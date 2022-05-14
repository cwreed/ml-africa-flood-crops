from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import date
import logging
import warnings

from src.exporters import CropHarvestExporter, CropHarvestSentinel2Exporter
from .base import BaseDataInstance, BaseEngineer, TestInstance

import numpy as np
import pandas as pd
import xarray as xr 

@dataclass
class CropHarvestDataInstance(BaseDataInstance):

    crop_label: int
    neighboring_array: np.ndarray

class CropHarvestEngineer(BaseEngineer):

    sentinel_dataset = CropHarvestSentinel2Exporter.dataset
    dataset = CropHarvestExporter.dataset
    label_str = 'is_crop'
    
    def read_labels(self, data_folder: Path) -> pd.DataFrame:
        cropharvest = data_folder / 'processed' / CropHarvestExporter.dataset / f"crop_labels_{self.region_name}.nc"
        assert cropharvest.exists(), "process_labels.py must be run to load labels"
        return xr.open_dataset(cropharvest).to_dataframe().dropna().reset_index()
    
    def calculate_ndvi(self, input_array: np.ndarray, n_dims: int=2) -> np.ndarray:
        """Calculates and adds NDVI as a band to the given input array"""
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
                'ignore', message='invalid value encountered in true_divide'
            )

            ndvi = np.where(
                (near_infrared - red) > 0,
                (near_infrared - red) / (near_infrared + red),
                0
            )
        
        return np.append(input_array, np.expand_dims(ndvi, -1), axis = -1)
    
    def process_single_file(
        self,
        filepath: Path,
        label_id: int,
        nan_fill: float,
        max_nan_ratio: float,
        calculate_normalizing_dict: bool,
        start_date: date,
        days_per_timestep: int,
        is_test: bool,
        sliding_window: bool=False,
        n_timesteps_per_instance: Optional[int]=None
    ) -> Optional[CropHarvestDataInstance]:

        logger = logging.getLogger(__name__)

        crop_label = int(self.labels.loc[label_id, self.label_str])

        da = self.load_tif(
            filepath=filepath, 
            days_per_timestep=days_per_timestep, 
            start_date=start_date, 
            n_timesteps_per_instance=None
        )

        min_lon, min_lat = float(da.x.min()), float(da.y.min())
        max_lon, max_lat = float(da.x.max()), float(da.y.max())

        overlap = self.labels.loc[
            (self.labels.lon <= max_lon) &
            (self.labels.lat <= max_lat) &
            (self.labels.lon >= min_lon) &
            (self.labels.lat >= min_lat)
        ]

        if len(overlap) == 0:
            return None

        """Select the first row of overlap for the label"""
        label_lon = overlap.iloc[0].lon
        label_lat = overlap.iloc[0].lat

        closest_lon = self.find_nearest(da.x, label_lon)
        closest_lat = self.find_nearest(da.y, label_lat)

        labeled_np = da.sel(x=closest_lon).sel(y=closest_lat).to_array().values[0]

        neighbor_lat, neighbor_lon = self.randomly_select_lat_lon(
            lat=da.y, lon=da.x, label_lat=label_lat, label_lon=label_lon
        )

        neighbor_np = da.sel(x=neighbor_lon).sel(y=neighbor_lat).to_array().values[0]

        """Calculate NDVI"""        
        labeled_np = self.calculate_ndvi(labeled_np)
        neighbor_np = self.calculate_ndvi(neighbor_np)

        labeled_array = self.fill_nan(array=labeled_np, nan_fill=nan_fill, max_ratio=max_nan_ratio)
        neighboring_array = self.fill_nan(array=neighbor_np, nan_fill=nan_fill)

        if (labeled_array is not None) & (not is_test) & calculate_normalizing_dict:
            self.update_normalizing_values(labeled_array)
        
        if labeled_array is not None:
            return CropHarvestDataInstance(
                crop_label=crop_label,
                label_lat=label_lat,
                label_lon=label_lon,
                instance_lat=closest_lat,
                instance_lon=closest_lon,
                labeled_array=labeled_array,
                neighboring_array=neighboring_array
            )
        else:
            logger.error("Too many NaN values--skipping")
            return None

    def tif_to_np(
        self,
        filepath: Path,
        start_date: date,
        add_ndvi: bool,
        nan_fill: float,
        days_per_timestep: int,
        sliding_window: bool,
        n_timesteps_per_instance: Optional[int]=None,
        normalizing_dict: Optional[dict[str, np.ndarray]]=None,
    ) -> list[TestInstance]:
        """Loads a region of data from a TIF into a TestInstance for predictions"""

        x = self.load_tif(
            filepath=filepath, 
            start_date=start_date,
            days_per_timestep=days_per_timestep,
            sliding_window=sliding_window,
            n_timesteps_per_instance=n_timesteps_per_instance
        )

        if isinstance(x, list):
            test_instances: list[TestInstance] = []

            for instance in x:
                lon, lat = np.meshgrid(instance.x.values, instance.y.values)
                flat_lon, flat_lat = (
                    np.squeeze(lon.reshape(-1, 1), -1),
                    np.squeeze(lat.reshape(-1, 1), -1)
                )

                x_np = instance.values[0]
                x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2]*x_np.shape[3])
                x_np = np.moveaxis(x_np, -1, 0)

                if add_ndvi:
                    x_np = self.calculate_ndvi(x_np, n_dims=3)

                x_np = self.fill_nan(x_np, nan_fill=nan_fill)

                if normalizing_dict is not None:
                    x_np = (x_np - normalizing_dict['mean']) / normalizing_dict['std']

                test_instances.append(TestInstance(x=x_np, lat=flat_lat, lon=flat_lon))
            
            return test_instances
        else:
            lon, lat = np.meshgrid(x.x.values, x.y.values)
            flat_lon, flat_lat = (
                np.squeeze(lon.reshape(-1, 1), -1),
                np.squeeze(lat.reshape(-1, 1), -1)
            )

            x_np = x.values[0]
            x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2]*x_np.shape[3])
            x_np = np.moveaxis(x_np, -1, 0)

            if add_ndvi:
                x_np = self.calculate_ndvi(x_np, n_dims=3)

            x_np = self.fill_nan(x_np, nan_fill=nan_fill)

            if normalizing_dict is not None:
                x_np = (x_np - normalizing_dict['mean']) / normalizing_dict['std']
            
            return [TestInstance(x=x_np, lat=flat_lat, lon=flat_lon)]
