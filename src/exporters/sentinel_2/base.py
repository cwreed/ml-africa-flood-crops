import os 
from pathlib import Path
from typing import Union
from abc import ABC, abstractmethod
from datetime import date, timedelta, datetime
import logging

from ..base import BaseExporter
from src.utils.regions import combine_bounding_boxes, STR2BB, REGIONS
from . import cloudfree

import ee
import pandas as pd

class BaseSentinel2Exporter(BaseExporter, ABC):

    ee_im_coll = 'COPERNICUS/S2_SR'
    min_date = date(2017, 3, 28)

    def __init__(self, data_folder: Path, region: Union[str, list[str]], combine_regions: bool=False) -> None:
        super().__init__(data_folder=data_folder)

        assert (
            (
                (isinstance(region, str)) &
                (
                    (region in REGIONS.keys()) | 
                    (region in STR2BB.keys())
                )
            ) |
            (
                (isinstance(region, list)) & 
                all(r in STR2BB.keys() for r in region)
            )
        ), f"Region must be one of {REGIONS.keys()} or one or more of {STR2BB.keys()}."
        

        self.region = region

        if combine_regions:
            self.region_type = 'single'
            self.region_bbox = combine_bounding_boxes(region)
        else:
            if (type(region) is str) & (region in REGIONS.keys()):
                self.region_type = 'multiple'
                self.region_bbox = [STR2BB[r] for r in REGIONS[region]]
            elif (type(region) is str) & (region in STR2BB.keys()):
                self.region_type = 'single'
                self.region_bbox = STR2BB[region]
            else:
                self.region_type = 'multiple'
                self.region_bbox = [STR2BB[r] for r in region]
        
        if self.region_type == 'multiple':
            self.region_name = "_".join(self.region).lower()
        else:
            self.region_name = self.region.lower()

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

        if checkpoint and os.path.exists(os.path.join(self.output_folder, f"{filename}.tif")):
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

