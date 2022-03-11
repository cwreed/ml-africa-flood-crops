import os
from typing import Union
from datetime import date
import logging

from .utils import EEBoundingBox, bounding_box_from_center, bounding_boxes_to_polygon
from .base import BaseExporter
from src.utils.regions import combine_bounding_boxes, STR2BB, REGIONS

import ee
import pandas as pd


class C2SDFOExporter(BaseExporter):

    dataset = 'cloud-to-street-dfo'
    ee_im_coll = 'GLOBAL_FLOOD_DB/MODIS_EVENTS/V1'

    def __init__(self, region: Union[str, list[str]], combine_regions: bool=False):
        super().__init__()

        assert (
            (
                (type(region) is str) &
                (
                    (region in REGIONS.keys()) | 
                    (region in STR2BB.keys())
                )
            ) |
            (
                (type(region) is list) & 
                all(r in STR2BB.keys() for r in region)
            ), 
                f"Region must be one of {REGIONS.keys()} or one or more of {STR2BB.keys()}."
        )

        self.region = region

        if combine_regions:
            self.region_type = 'single'
            self.region_bbox = combine_bounding_boxes(region)
            self.ee_region_geo = EEBoundingBox(self.region_bbox).to_ee_polygon()
        else:
            if (type(region) is 'str') & (region in REGIONS.keys()):
                self.region_type = 'multiple'
                self.region_bbox = [STR2BB[r] for r in REGIONS[region]]
                self.ee_region_geo = bounding_boxes_to_polygon([EEBoundingBox(region) for region in self.region_bbox])
            elif (type(region) is 'str') & (region in STR2BB.keys()):
                self.region_type = 'single'
                self.region_bbox = STR2BB[region]
                self.ee_region_geo = EEBoundingBox(self.region_bbox).to_ee_polygon()
            else:
                self.region_type = 'multiple'
                self.region_bbox = [STR2BB[r] for r in region]
                self.ee_region_geo = bounding_boxes_to_polygon([EEBoundingBox(region) for region in self.region_bbox])
        
    def find_positive_labels(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        This function locates flooded pixels within the class region and between provided start and end dates
        and returns them as a pandas DataFrame with columns for lon, lat, flooded indicator, and other
        labels provided by the Cloud-to-Street/DFO dataset
        """
        c2sdfo = ee.ImageCollection(self.ee_im_coll)

        flood_mask = lambda img: img.select('flooded').gt(0).And(img.select('jrc_perm_water').eq(0))

        positive_floods = (c2sdfo
            .filterBounds(self.ee_region_geo)
            .filterDate(ee.Date(start_date), ee.Date(end_date))
            .map(lambda img: img.addBands(img.pixelLonLat()))
            .map(lambda img: img.updateMask(flood_mask(img)))
        )

        positive_labels_df = pd.DataFrame(positive_floods.iterate(self.img_to_data_dict))

        return positive_labels_df

    def find_negative_labels(
            self, 
            positive_labels: pd.DataFrame, 
            start_date: date, 
            end_date: date, 
            n: int
        ) -> pd.DataFrame:
        """
        This function locates `n` negatively labeled pixels within the class region and between provided start and end dates
        and returns them as a pandas DataFrame with columns for lon, lat, flooded indicator, and other
        labels provided by the Cloud-to-Street/DFO dataset.

        Types of negative pixels:
            - Permanent water bodies
            - Negative pixels adjacent to positive pixels
            - Negative pixels not near positive pixels
        """
        c2sdfo = ee.ImageCollection(self.ee_im_coll)

        non_flood_mask = lambda img: img.select('flooded').eq(0).And(img.select('jrc_perm_water').eq(0))
        perm_water_mask = lambda img: img.select('jrc_perm_water').gt(0)

        """Get permanent water labels"""
        perm_water = (c2sdfo
            .filterBounds(self.ee_region_geo)
            .filterDate(ee.Date(start_date), ee.Date(end_date))
            .map(lambda img: img.addBands(img.pixelLonLat()))
            .map(lambda img: img.updateMask(perm_water_mask(img)))
        )

        perm_water_df = pd.DataFrame(perm_water.iterate(self.img_to_data_dict))

        """Get negative flood near positive pixel labels"""
        positive_bboxes = positive_labels.apply(lambda x: bounding_box_from_center(x.latitude, x.longitude, 500), axis = 1)
        positive_ee_geo = ee.Geometry.MultiPolygon([bbox.to_ee_polygon for bbox in positive_bboxes])

        negative_floods_near_positive = (c2sdfo
            .filterBounds(positive_ee_geo)
            .filterDate(ee.Date(start_date), ee.Date(end_date))
            .map(lambda img: img.addBands(img.pixelLonLat()))
            .map(lambda img: img.updateMask(non_flood_mask(img)))
        )

        negative_floods_near_positive_df = pd.DataFrame(negative_floods_near_positive.iterate(self.img_to_data_dict))

        """Get negative flood labels from elsewhere"""
        negative_ee_geo = self.ee_region_geo.difference(positive_ee_geo)

        negative_floods_not_near_positive = (c2sdfo
            .filterBounds(negative_ee_geo)
            .filterDate(ee.Date(start_date), ee.Date(end_date))
            .map(lambda img: img.addBands(img.pixelLonLat()))
            .map(lambda img: img.updateMask(non_flood_mask(img)))
        )

        negative_floods_not_near_positive_df = pd.DataFrame(negative_floods_not_near_positive.iterate(self.img_to_data_dict))

        """Sample an equal number of labels from each"""
        perm_water_samples = perm_water_df.sample(n = round(n/3))
        negative_near_samples = negative_floods_near_positive_df.sample(n = round(n/3))
        negative_not_near_samples = negative_floods_not_near_positive_df.sample(n = round(n/3))

        negative_samples_df = pd.concat([perm_water_samples, 
                                      negative_near_samples, 
                                      negative_not_near_samples], axis = 0)

        """Clean up the unpassed variables"""
        del (
            perm_water_df, 
            negative_floods_near_positive_df,
            negative_floods_not_near_positive_df,
            perm_water_samples, 
            negative_near_samples,
            negative_not_near_samples,
            positive_bboxes,
            positive_ee_geo,
            negative_ee_geo
        )

        return negative_samples_df

    
    def img_to_data_dict(self, img: ee.Image, first: list[dict]=[]) -> list[dict]:
        """
        Function to be used with Earth Engine's .iterate() to turn an
        ee.ImageCollection into a list of dictionaries.
        """
        data_dict = (
            img
            .reduceRegion(
                ee.Reducer.toList(),
                self.ee_region_geo
            )
        )

        n = len(data_dict[data_dict.keys()[0]])

        data_dict['began'] = [img.get('began').getInfo()] * n
        data_dict['ended'] = [img.get('ended').getInfo()] * n

        return first.append(data_dict)
        
    def export(self, start_date: date, end_date: date) -> None:
        """
        Download the Cloud-to-Street/DFO labels and store in the output folder
        """

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        logger.info("Beginning export of Cloud-to-Street/DFO flood labels")
        logger.info("Finding positive labels")
        positive_labels = self.find_positive_labels(start_date, end_date)

        logger.info("Finding negative labels")
        negative_labels = self.find_negative_labels(positive_labels, start_date, end_date, positive_labels.shape[0])

        flood_labels = pd.concat([positive_labels, negative_labels], axis = 0)

        if self.region_type == 'multiple':
            region_name = "_".join(self.region)
        else:
            region_name = self.region

        outpath = os.path.join(
            self.output_folder, 
            f"{region_name}_{start_date.month}{start_date.year}_{end_date.month}{end_date.year}.csv"
        )

        logger.info(f"Writing flood labels to {outpath}")
        flood_labels.to_csv(outpath)