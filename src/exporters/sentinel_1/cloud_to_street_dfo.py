import os
from pathlib import Path
from typing import Union
from datetime import date, timedelta

from .base import BaseSentinel1Exporter
from ..cloud_to_street_dfo import C2SDFOExporter
from ..utils import EEBoundingBox, bounding_box_from_center
from src.utils.regions import REGIONS, STR2BB

import ee
import numpy as np
import pandas as pd
import xarray as xr

class C2SDFOSentinel1Exporter(BaseSentinel1Exporter):
    
    dataset = 'cloud-to-street-dfo-sentinel-1'

    def load_labels(self) -> pd.DataFrame:
        label_file = os.path.join(
            self.data_folder,
            "processed",
            C2SDFOExporter.dataset,
            f"flood_labels_{self.region_name}.nc"
        )

        assert (
            os.path.exists(label_file)
        ), f"process_labels.py must be run in order to read in {label_file}"

        return xr.open_dataset(label_file).to_dataframe().dropna().reset_index()

        
    def labels_to_bounding_boxes(self, surrounding_meters: int) -> list[EEBoundingBox]:
        
        ee_bboxes = self.labels.apply(
            lambda row: bounding_box_from_center(row.lat, row.lon, surrounding_meters=surrounding_meters),
            axis = 1
        ).tolist()

        return ee_bboxes

    def export_for_labels(
        self,
        days_per_timestep: int=12,
        n_timesteps: int=4,
        surrounding_meters: int = 80,
        checkpoint: bool = True,
        monitor: bool = False
    ) -> None:
        """Exports Sentinel-1 data for labels in self.labels"""

        flood_end_dates = pd.to_datetime(self.labels.ended)
        flood_start_dates = pd.to_datetime(self.labels.began)

        input_start_dates = flood_start_dates - timedelta(days = days_per_timestep * n_timesteps)


        """Figure out whether there is Sentinel-1 data to pull for dates"""
        min_date = np.min(input_start_dates)
        n_with_timestamp_deficit = np.sum(input_start_dates < np.datetime64(self.min_date))

        assert (
            min_date >= np.datetime64(self.min_date)
        ), (
            f"Sentinel-2 data is not available prior to {self.min_date}\n"
            f"Unable to acquire {n_timesteps} timesteps of data for {n_with_timestamp_deficit} samples in labels set.\n"
            f"Consider updating label dates or setting `override_label_dates` arg to True."
        )

        label_bounding_boxes = self.labels_to_bounding_boxes(surrounding_meters=surrounding_meters)

        for i, bbox in enumerate(label_bounding_boxes):
            self.export_for_polygon(
                polygon=bbox.to_ee_polygon(),
                polygon_identifier=i,
                start_date=input_start_dates[i],
                end_date=flood_end_dates[i],
                checkpoint=checkpoint,
                monitor=monitor
            )





