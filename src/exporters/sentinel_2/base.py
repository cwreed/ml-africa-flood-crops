from pathlib import Path
from typing import Union
from abc import ABC, abstractmethod
from datetime import date, timedelta, datetime
import logging

from ..base import BaseExporter
from src.utils.regions import _initialize_regions
from src.utils.sentinel import SENTINEL_2_BANDS, SENTINEL_2_START_DATE
from . import cloudfree

import ee
import pandas as pd

class BaseSentinel2Exporter(BaseExporter, ABC):

    ee_im_coll = 'COPERNICUS/S2_SR'
    BANDS = SENTINEL_2_BANDS
    min_date = SENTINEL_2_START_DATE

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
            days_per_timestep: int,
            checkpoint: bool,
            monitor: bool
        ) -> None:
        """
        Function pulled from https://github.com/nasaharvest/togo-crop-mask/blob/61da13504faf085b99ddea3d38aadd0b7baecee4/src/exporters/sentinel/base.py
        """
        logger = logging.getLogger(__name__)

        curr_start_date = start_date
        curr_end_date = curr_start_date + timedelta(days=days_per_timestep)

        image_collection_list: list[ee.Image] = []

        logger.info(
            f"Exporting image for polygon {polygon_identifier} from "
            f"aggregated images between {datetime.strftime(start_date, '%Y-%m-%d')} and {datetime.strftime(end_date, '%Y-%m-%d')}."
        )

        filename = f"{polygon_identifier}_{datetime.strftime(start_date, '%Y%m%d')}_{datetime.strftime(end_date, '%Y%m%d')}"

        if checkpoint and (self.output_folder / f"{filename}.tif").exists():
            logger.info(f"{filename}.tif already exists--skipping")
            return None

        while curr_end_date <= end_date:
            image_collection_list.append(
                cloudfree.get_single_image(
                    region=polygon, start_date=curr_start_date, end_date=curr_end_date
                )
            )

            curr_start_date += timedelta(days=days_per_timestep)
            curr_end_date += timedelta(days=days_per_timestep)

        imcoll = ee.ImageCollection(image_collection_list)

        """Combine images into a single image"""
        img = ee.Image(imcoll.iterate(cloudfree.combine_bands))

        cloudfree.export(
            image=img,
            region=polygon,
            filename=filename,
            drive_folder=self.dataset,
            monitor=monitor
        )

