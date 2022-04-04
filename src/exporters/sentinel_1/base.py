from pathlib import Path
from typing import Union
from abc import ABC, abstractmethod
from datetime import date, datetime
import logging

from ..base import BaseExporter
from src.utils.regions import _initialize_regions
from src.utils.sentinel import SENTINEL_1_BANDS, SENTINEL_1_START_DATE

import ee
import pandas as pd

class BaseSentinel1Exporter(BaseExporter, ABC):

    ee_im_coll = 'COPERNICUS/S1_GRD'
    BANDS = SENTINEL_1_BANDS
    min_date = SENTINEL_1_START_DATE

    def __init__(self, data_folder: Path, region: Union[str, list[str]], combine_regions: bool=False) -> None:
        super().__init__(data_folder=data_folder)
        
        _initialize_regions(self, region=region, combine_regions=combine_regions)

        self.labels = self.load_labels()

    @abstractmethod
    def load_labels(self) -> pd.DataFrame:
        raise NotImplementedError

    def export_for_polygon(
            self,
            polygon: ee.Geometry.Polygon,
            polygon_identifier: Union[int, str],
            start_date: date,
            end_date: date,
            checkpoint: bool,
            monitor: bool
        ) -> None:
        """
        Function inpsired by https://github.com/nasaharvest/togo-crop-mask/blob/61da13504faf085b99ddea3d38aadd0b7baecee4/src/exporters/sentinel/base.py
        """

        logger = logging.getLogger(__name__)

        logger.info(
            f"Exporting image for polygon {polygon_identifier} from "
            f"aggregated images between {datetime.strftime(start_date, '%Y-%m-%d')} and {datetime.strftime(end_date, '%Y-%m-%d')}."
        )

        filename = f"{polygon_identifier}_{datetime.strftime(start_date, '%Y%m%d')}_{datetime.strftime(end_date, '%Y%m%d')}"

        if checkpoint and (self.output_folder / f"{filename}.tif").exists():
            logger.info(f"{filename} already exists--skipping")
            return None
        
        imcoll = (
            ee.ImageCollection(self.ee_im_coll)
            .filterBounds(polygon)
            .filterDate(ee.Date(start_date), ee.Date(end_date))
        )

        """Combine images into a single image"""
        img = ee.Image(imcoll.iterate(self.combine_bands))

        self.export_image(
            image=img,
            region=polygon,
            filename=filename,
            drive_folder=self.dataset,
            monitor=monitor
        )

    def combine_bands(self, current: ee.Image, previous: ee.Image):
        """
        Transforms an Image Collection with 1 band per Image into a single Image with items as bands
        Author: Jamie Vleeshouwer
        """
        previous = ee.Image(previous)
        current = current.select(self.BANDS)
        
        return ee.Algorithms.If(
            ee.Algorithms.IsEqual(previous, None),
            current,
            previous.addBands(ee.Image(current)),
        )
    
    def export_image(
            self,
            image: ee.Image,
            region: ee.Geometry,
            filename: str,
            drive_folder: str,
            monitor: bool = False
        ) -> ee.batch.Export:
        
        task = ee.batch.Export.image(
            image.clip(region),
            filename,
            {"scale": 10, "region": region, "maxPixels": 1e13, "driveFolder": drive_folder, "crs": 'EPSG:4326'}
        )

        try:
            task.start()
        except ee.ee_exception.EEException as e:
            print(f"Task not started! Got exception {e}")
            return task

        if monitor:
            self.monitor_task(task)

        return task

    def monitor_task(self, task: ee.batch.Export) -> None:

        logger = logging.getLogger(__name__)

        while task.status()["state"] in ["READY", "RUNNING"]:
            logger.debug(task.status())
        

