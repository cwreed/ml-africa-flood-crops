from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import date
import logging

from src.exporters import CropHarvestExporter, CropHarvestSentinel2Exporter
from .base import BaseDataInstance, BaseEngineer

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
    
    def process_single_file(
        self,
        filepath: Path,
        label_id: int,
        nan_fill: float,
        max_nan_ratio: float,
        calculate_normalizing_dict: bool,
        start_date: date,
        days_per_timestep: int,
        is_test: bool
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
