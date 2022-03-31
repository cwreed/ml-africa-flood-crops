import os
from pathlib import Path
from datetime import date, timedelta
import logging

from .base import BaseSentinel2Exporter
from src.exporters import CropHarvestExporter
from src.exporters.utils import EEBoundingBox, bounding_box_from_center

import numpy as np
import pandas as pd
import xarray as xr


class CropHarvestSentinel2Exporter(BaseSentinel2Exporter):
    
    dataset = 'cropharvest-sentinel-2'

    def load_labels(self) -> pd.DataFrame:

        label_file = os.path.join(
            self.data_folder, 
            'processed', 
            CropHarvestExporter.dataset,
            f'crop_labels_{self.region_name}.nc'
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
            days_per_timestep: int,
            n_timesteps: int,
            surrounding_meters: int = 80,
            override_label_dates: bool = True,
            checkpoint: bool = True,
            monitor: bool = False
        ) -> None:
        """
        Creates a bounding box around each label, then exports historical cloud-free Sentinel-2 imagery for
        `n_timesteps` up to the date associated with the label. 

        `override_label_dates` can be used to reset the label dates such that the desired amount of data is available
        from Sentinel-2. This option should be used with caution, especially if the temporal aspect of when the label
        was sampled is important.
        
        """

        logger = logging.getLogger(__name__)

        end_dates = pd.to_datetime(self.labels.collection_date)
        start_dates = end_dates - timedelta(days = days_per_timestep * n_timesteps)

        """Figure out whether there is Sentinel-2 data to pull for dates"""
        min_date = np.min(start_dates)
        n_with_timestamp_deficit = np.sum(start_dates < np.datetime64(self.min_date))

        if override_label_dates:
            while (n_with_timestamp_deficit > 0):
                start_dates = pd.to_datetime(np.where(start_dates < np.datetime64(self.min_date), start_dates + timedelta(days = 365), start_dates))
                end_dates = start_dates + timedelta(days = days_per_timestep * n_timesteps)
                min_date = np.min(start_dates)
                n_with_timestamp_deficit = np.sum(start_dates < np.datetime64(self.min_date))

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
                start_date=start_dates[i],
                end_date=end_dates[i],
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor
            )








