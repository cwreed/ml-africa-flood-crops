from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import date, datetime
import logging

from src.exporters import C2SDFOExporter, C2SDFOSentinel1Exporter
from .base import BaseDataInstance, BaseEngineer

import numpy as np
import pandas as pd
import xarray as xr 

@dataclass
class C2SDFODataInstance(BaseDataInstance):

    flood_label: int
    began: date
    ended: date
    neighboring_array: np.ndarray

class C2SDFOEngineer(BaseEngineer):

    sentinel_dataset = C2SDFOSentinel1Exporter.dataset
    dataset = C2SDFOExporter.dataset

    def read_labels(self, data_folder: Path) -> pd.DataFrame:
        c2sdfo = data_folder / 'processed' / C2SDFOExporter.dataset / f"flood_labels_{self.region_name}.nc"
        assert c2sdfo.exists(), "process_labels.py must be run to load labels"
        return xr.open_dataset(c2sdfo).to_dataframe().dropna().reset_index()

    def process_single_file(
        self,
        filepath: Path,
        nan_fill: float,
        max_nan_ratio: float,
        calculate_normalizing_dict: bool,
        start_date: date,
        days_per_timestep: int,
        is_test: bool
    ) -> Optional[C2SDFODataInstance]:

        logger = logging.getLogger(__name__)

        da = self.load_tif(
            filepath=filepath, days_per_timestep=days_per_timestep, start_date=start_date
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

        labeled_np = da.sel(x=closest_lon).sel(y=closest_lat).values

        neighbor_lat, neighbor_lon = self.randomly_select_lat_lon(
            lat=da.y, lon=da.x, label_lat=label_lat, label_lon=label_lon
        )

        neighbor_np = da.sel(x=neighbor_lon).sel(y=neighbor_lat)

        labeled_array = self.fill_nan(labeled_np, nan_fill=nan_fill, max_ratio=max_nan_ratio)
        neighboring_array = self.fill_nan(neighbor_np, nan_fill=nan_fill)

        if (not is_test) & calculate_normalizing_dict:
            self.update_normalizing_values(labeled_array)
        
        if labeled_array is not None:
            return C2SDFODataInstance(
                label_lat=label_lat,
                label_lon=label_lon,
                instance_lat=closest_lat,
                instance_lon=closest_lon,
                labeled_array=labeled_array,
                neighboring_array=neighboring_array
            )
        else:
            logger.info("Too many NaN values--skipping")
            return None