import os
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
from datetime import date, datetime, timedelta
import logging
import warnings
from tqdm import tqdm
import pickle

from src.utils.regions import BoundingBox, _initialize_regions
from src.exporters.sentinel_2.base import BaseSentinel2Exporter
from src.exporters.sentinel_1.base import BaseSentinel1Exporter

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class BaseDataInstance:
    label_lat: float
    label_lon: float
    instance_lat: float
    instance_lon: float
    labeled_array: np.ndarray

    def isin(self, bounding_box: BoundingBox) -> bool:
        return (
            (self.instance_lon <= bounding_box.max_lon) &
            (self.instance_lon >= bounding_box.min_lon) &
            (self.instance_lat <= bounding_box.max_lat) &
            (self.instance_lat >= bounding_box.min_lat)
        )

class BaseEngineer(ABC):
    """
    Combines satellite imagery data from Sentinel 1 or 2 with their associated labels
    into numpy arrays for machine learning.

    Almost all functions are replicated from 
    https://github.com/nasaharvest/togo-crop-mask/blob/master/src/engineer/base.py


    """

    sentinel_dataset: str
    dataset: str

    def __init__(self, data_folder: Path, region: Union[str, list[str]], combine_regions: bool):
        _initialize_regions(self, region=region, combine_regions=combine_regions)

        self.data_folder = data_folder
        self.savedir = self.data_folder / 'features' / self.dataset
        self.savedir.mkdir(exists_ok=True, parents=True)

        self.geospatial_files = self.get_geospatial_files(data_folder)
        self.labels = self.read_labels(data_folder)

        if 'sentinel-2' in self.sentinel_dataset:
            self.BANDS = BaseSentinel2Exporter.BANDS
        else:
            self.BANDS = BaseSentinel1Exporter.BANDS

        self.normalizing_dict_interim: dict[str, Union[np.array, int]] = {"n": 0}


    def get_geospatial_files(self, data_folder: Path) -> list[Path]:
        sentinel_files = data_folder / 'raw' / self.sentinel_dataset
        return list(sentinel_files.glob('*.tif'))

    @staticmethod
    @abstractmethod
    def read_labels(data_folder: Path) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def find_nearest(array: xr.DataArray, value: float) -> np.array:
        array = np.asarray(array)
        idx = np.argmin(np.abs(array - value))
        return array[idx]

    @staticmethod
    def randomly_select_lat_lon(
        lat: np.ndarray,
        lon: np.ndarray,
        label_lat: float,
        label_lon: float
    ) -> tuple[float, float]:
        
        lats = np.random.choice(lat, size=2, replace=False)
        lons = np.random.choice(lon, size=2, replace=False)

        if (lats[0] != label_lat) or (lons[0] != label_lon):
            return lats[0], lons[0]
        else:
            return lats[1], lons[1]

    @staticmethod
    def process_filename(
        filename: str
    ) -> tuple[int, datetime, datetime]:

        date_format = '%Y%m%d'

        idx_str, start_date_str, end_date_str = filename[:-4].split("_")
        
        idx = int(idx_str)
        start_date = datetime.strptime(start_date_str, date_format)
        end_date = datetime.strptime(end_date_str, date_format)

        return idx, start_date, end_date


    def load_tif(
        self,
        filepath: Path,
        start_date: datetime,
        days_per_timestep: int
    ) -> xr.DataArray:
        
        scaling_factor = 10000 if 'sentinel-2' in self.sentinel_dataset else 1
        
        da = xr.open_dataset(filepath, engine='rasterio').rename({'band_data': 'FEATURES'}) / scaling_factor

        da_split_by_time: list[xr.DataArray] = []

        bands_per_timestep = len(self.BANDS)
        num_bands = len(da.band)

        assert (
            num_bands % bands_per_timestep == 0
        ), "Total number of bands not divisible by the expected bands per timestep!"

        cur_band = 0
        while cur_band + bands_per_timestep <= num_bands:
            time_specific_da = da.isel(
                band=slice(cur_band, cur_band+bands_per_timestep)
            )

            da_split_by_time.append(time_specific_da)
            cur_band += bands_per_timestep
        
        timesteps = [
            start_date + timedelta(days=days_per_timestep) * i
            for i in range(len(da_split_by_time))
        ]

        combined = xr.concat(da_split_by_time, pd.Index(timesteps, name='time'))
        combined.attrs['band_descriptions'] = self.BANDS

        return combined

    def update_normalizing_values(self, array: np.ndarray) -> None:
        
        num_bands = array.shape[1]

        if "mean" not in self.normalizing_dict_interim:
            self.normalizing_dict_interim['mean'] = np.zeros(num_bands)
            self.normalizing_dict_interim['M2'] = np.zeros(num_bands)

        for time_idx  in range(array.shape[0]):
            self.normalizing_dict_interim['n'] += 1
            x = array[time_idx, :]

            delta = x - self.normalizing_dict_interim['mean']
            self.normalizing_dict_interim['mean'] += (
                delta / self.normalizing_dict_interim['n']
            )
            self.normalizing_dict_interim['M2'] += delta * (
                x - self.normalizing_dict_interim['mean']
            )

    def calculate_normalizing_dict(self) -> Optional[dict[str, np.ndarray]]:
        
        logger = logging.getLogger(__name__)

        if "mean" not in self.normalizing_dict_interim:
            logger.error(
                "No normalizing dict calculated--make sure to call `update_normalizing_values`"
            )
            return None
        
        variance = self.normalizing_dict_interim['M2'] / (
            self.normalizing_dict_interim['n'] - 1
        )

        std = np.sqrt(variance)

        return {'mean': self.normalizing_dict_interim['mean'], 'std': std}

    def fill_nan(array: np.ndarray, nan_fill: float, max_ratio: Optional[float] = None) -> Optional[np.ndarray]:
        logger = logging.getLogger(__name__)

        logger.info("Filling NaN values in data array")

        if max_ratio is not None:
            n_nan = np.count_nonzero(np.isnan(array))
            if (n_nan / array.size) > max_ratio:
                logger.debug("Number of NaN values exceeds quota set by `max_ratio`, returning None")
                return None
        
        return np.nan_to_num(array, nan=nan_fill)
    
    @abstractmethod
    def process_single_file(self) -> Optional[BaseDataInstance]:
        raise NotImplementedError
    
    def calculate_ndvi(self, input_array: np.ndarray, n_dims: int=2) -> np.ndarray:

        assert (
            'sentinel-2' in self.sentinel_dataset
        ), f"Can only calculate NDVI for Sentinel-2 datasets: current dataset is {self.sentinel_dataset}"

        if n_dims == 2:
            near_infrared = input_array[:, self.BANDS.index('B8')]
            red = input_array[:, self.BANDS.index('B4')]
        elif n_dims == 3:
            near_infrared = input_array[:, :, self.BANDS.index('B8')]
            red = input_array[:, :, self.BANDS.index('B4')]
        else:
            raise ValueError(f"Expected n_dims to be 2 or 3: got {n_dims}")
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='invalude encountered in true_divide'
            )

            ndvi = np.where(
                (near_infrared - red) > 0,
                (near_infrared - red) / (near_infrared + red),
                0
            )
        
        return np.append(input_array, np.expand_dims(ndvi, -1), axis = -1)

    def engineer(
        self,
        val_set_size: float=0.1,
        test_set_size: float=0.2,
        nan_fill: float=0.,
        max_nan_ratio: float=0.3,
        checkpoint: bool=True,
        add_ndvi: bool=True,
        calculate_normalizing_dict: bool=True,
        days_per_timestep: int=30
    ) -> None:
        """
        Engineer the exported/processed Earth Engine data and labels into 
        train/validation/test numpy arrays for ML
        """
        logger = logging.getLogger(__name__)

        for filepath in tqdm(self.geospatial_files):
            identifier, start_date, end_date = self.process_filename(filepath.name)

            filename = f"{identifier}_{datetime.strptime(start_date, '%Y%m%d')}_{datetime.strptime(end_date, '%Y%m%d')}"

            if checkpoint:
                """Check to see if files have already been written"""

                if (
                    (self.savedir / 'train' / f"{filename}.pkl").exists() |
                    (self.savedir / 'validation' / f"{filename}.pkl").exists() |
                    (self.savedir / 'test' / f"{filename}.pkl").exists() 
                ):
                    logger.info(f'{filepath.name} has already been engineered--skipping!')
                    continue
            
            random_float = np.random.uniform()

            if random_float <= (val_set_size + test_set_size):
                if random_float <= val_set_size:
                    data_subset = 'validation'
                else:
                    data_subset = 'test'
            else:
                data_subset = 'train'
            
            instance = self.process_single_file(
                filepath,
                nan_fill=nan_fill,
                max_nan_ratio=max_nan_ratio,
                add_ndvi=add_ndvi,
                calculated_normalizing_dict=calculate_normalizing_dict,
                start_date=start_date,
                days_per_timestep=days_per_timestep,
                is_test=True if data_subset == 'test' else False
            )

            if instance is not None:
                subset_path = self.savedir / data_subset
                subset_path.mkdir(exists_ok=True)
                save_path = subset_path / f"{filename}.pkl"
                with save_path.open('wb') as f:
                    pickle.dump(instance, f)
            
            if calculate_normalizing_dict:
                normalizing_dict = self.calculate_normalizing_dict()

                if normalizing_dict is not None:
                    save_path = self.savedir / 'normalizing_dict.pkl'
                    with save_path.open('wb') as f:
                        pickle.dump(normalizing_dict, f)
                else:
                    logger.debug("No normalizing dict calculated!")
                  






            






