from pathlib import Path
from typing import Optional
from datetime import date, timedelta
import logging

from .base import BaseSentinel1Exporter
from ..utils import EEBoundingBox
from src.utils.regions import BoundingBox, STR2BB, REGIONS

import pandas as pd
import geopandas as gpd

class RegionalSentinel1Exporter(BaseSentinel1Exporter):
    """
    Exports entire regions of Sentinel-1 data to use for inference and mapping
    with a trained FloodMapper model.
    """

    dataset = 'earth-engine-region-sentinel-1-big'

    def load_labels(self) -> pd.DataFrame:
        return pd.DataFrame()

    def load_region_geojson(self, boundary_file: str) -> EEBoundingBox:
        boundary_file = (
            self.data_folder /
            'raw' /
            'prediction-boundaries' /
            boundary_file
        )

        assert boundary_file.exists(), f"Could not find boundary file to load at {boundary_file}"

        boundary = gpd.read_file(boundary_file, driver='GEOJSON')

        bounds = boundary.bounds
        min_lon = bounds.minx.values[0]
        min_lat = bounds.miny.values[0]
        max_lon = bounds.maxx.values[0]
        max_lat = bounds.maxy.values[0]

        return EEBoundingBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)

    def export_for_region(
        self,
        inference_start_date: date,
        inference_end_date: date,
        days_per_timestep: int=12,
        n_timesteps: int=6,
        region_boundary_file: Optional[str]=None,
        checkpoint: bool=True,
        monitor: bool=False,
    ):
        if region_boundary_file is not None:
            export_region = self.load_region_geojson(region_boundary_file)
            export_region_name = region_boundary_file[:-8]
        else:
            export_region = EEBoundingBox(bbox=STR2BB[self.region])
            export_region_name = self.region
        
        input_start_date = inference_start_date - timedelta(days = days_per_timestep * n_timesteps)

        self.export_for_polygon(
            polygon=export_region.to_ee_polygon(),
            polygon_identifier=export_region_name,
            start_date=input_start_date,
            end_date=inference_end_date,
            checkpoint=checkpoint,
            monitor=monitor
        )

        
