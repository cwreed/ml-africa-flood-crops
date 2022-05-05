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
    flood_prior: int
    perm_water: int
    neighboring_array: np.ndarray

class C2SDFOEngineer(BaseEngineer):

    sentinel_dataset = C2SDFOSentinel1Exporter.dataset
    dataset = C2SDFOExporter.dataset
    label_str = 'flooded'
    perm_water_str = 'jrc_perm_water'

    def read_labels(self, data_folder: Path) -> pd.DataFrame:
        """Read in the processed labels"""
        c2sdfo = data_folder / 'processed' / C2SDFOExporter.dataset / f"flood_labels_{self.region_name}.nc"

        assert c2sdfo.exists(), "process_labels.py must be run to load labels"

        label_df = xr.open_dataset(c2sdfo).to_dataframe().dropna().reset_index()

        label_df['began'] = pd.to_datetime(label_df['began']).dt.date
        label_df['ended'] = pd.to_datetime(label_df['ended']).dt.date

        return label_df

    def process_single_file(
        self,
        filepath: Path,
        label_id: int,
        nan_fill: float,
        max_nan_ratio: float,
        calculate_normalizing_dict: bool,
        start_date: date,
        days_per_timestep: int,
        sliding_window: bool,
        n_timesteps_per_instance: Optional[int],
        is_test: bool
    ) -> Optional[C2SDFODataInstance]:

        logger = logging.getLogger(__name__)

        """Get label details"""
        flood_label = self.labels.loc[label_id, self.label_str]
        perm_water_label = self.labels.loc[label_id, self.perm_water_str]
        began = self.labels.loc[label_id, 'began']
        ended = self.labels.loc[label_id, 'ended']

        """Now engineer the geospatial data"""
        da_windows = self.load_tif(
            filepath=filepath, 
            days_per_timestep=days_per_timestep, 
            start_date=start_date, 
            sliding_window=sliding_window,
            n_timesteps_per_instance=n_timesteps_per_instance,
        )

        data_instances: list[C2SDFODataInstance] = []

        for da in da_windows:

            """Determine whether the end of the data window falls outside of the flooded period in time"""

            inference_time = da.time.max()
            before_flood = (inference_time < np.datetime64(began))
            after_flood = (inference_time > np.datetime64(ended))

            """Now engineer the arrays"""

            min_lon, min_lat = float(da.x.min()), float(da.y.min())
            max_lon, max_lat = float(da.x.max()), float(da.y.max())

            overlap = self.labels.loc[
                (self.labels.lon <= max_lon) &
                (self.labels.lat <= max_lat) &
                (self.labels.lon >= min_lon) &
                (self.labels.lat >= min_lat)
            ]

            if len(overlap) == 0:
                continue

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

            labeled_array = self.fill_nan(labeled_np, nan_fill=nan_fill, max_ratio=max_nan_ratio)
            neighboring_array = self.fill_nan(neighbor_np, nan_fill=nan_fill)

            if (labeled_array is not None) & (not is_test) & calculate_normalizing_dict:
                self.update_normalizing_values(labeled_array)
            
            if labeled_array is not None:
                data_instances.append(
                    C2SDFODataInstance(
                        flood_label=0 if (before_flood | after_flood) else flood_label,
                        flood_prior=flood_label if after_flood else 0,
                        perm_water=perm_water_label,                  
                        label_lat=label_lat,
                        label_lon=label_lon,
                        instance_lat=closest_lat,
                        instance_lon=closest_lon,
                        labeled_array=labeled_array,
                        neighboring_array=neighboring_array
                    )
                )
            else:
                logger.error("Too many NaN values--skipping")
                continue
        
        return data_instances