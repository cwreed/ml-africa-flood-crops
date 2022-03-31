import os
from pathlib import Path
from typing import Union
from datetime import date
import logging

from .utils import EEBoundingBox, bounding_box_from_center, bounding_boxes_to_polygon
from .base import BaseExporter
from src.utils.regions import combine_bounding_boxes, STR2BB, REGIONS

import ee
import pandas as pd
import numpy as np


class C2SDFOExporter(BaseExporter):

    dataset = 'cloud-to-street-dfo'
    ee_im_coll = 'GLOBAL_FLOOD_DB/MODIS_EVENTS/V1'

    def __init__(self, data_folder: Path, region: Union[str, list[str]], combine_regions: bool=False):
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
            self.ee_region_geo = EEBoundingBox(self.region_bbox).to_ee_polygon()
        else:
            if (isinstance(region, str)) & (region in REGIONS.keys()):
                self.region_type = 'multiple'
                self.region_bbox = [STR2BB[r] for r in REGIONS[region]]
                self.ee_region_geo = bounding_boxes_to_polygon([EEBoundingBox(region) for region in self.region_bbox])
            elif (isinstance(region, str)) & (region in STR2BB.keys()):
                self.region_type = 'single'
                self.region_bbox = STR2BB[region]
                self.ee_region_geo = EEBoundingBox(self.region_bbox).to_ee_polygon()
            else:
                self.region_type = 'multiple'
                self.region_bbox = [STR2BB[r] for r in region]
                self.ee_region_geo = bounding_boxes_to_polygon([EEBoundingBox(region) for region in self.region_bbox])

        self.random_seed = np.random.default_rng(2022)
        
    def find_positive_labels(
            self, 
            start_date: date, 
            end_date: date,
            n: int
        ) -> pd.DataFrame:
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

        
        positive_labels_df = self.dicts_to_dataframe(
                                    positive_floods
                                    .iterate(
                                        self.reduce_img_to_data_dict, 
                                        first=ee.List([])
                                    )
                                    .getInfo()
                                ).sample(n, random_state=self.random_seed)

        return positive_labels_df

    def find_negative_labels(
            self, 
            positive_labels: pd.DataFrame, 
            start_date: date, 
            end_date: date, 
            negative_to_positive_ratio: int,
            logger: logging.Logger
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

        """Calculate the number of negative samples to take per image"""

        self.n_neg_per_img = (positive_labels.groupby(['began', 'ended']).size().values * negative_to_positive_ratio).tolist()
        logger.info(f"Number of negative samples that will be taken per image = {self.n_neg_per_img}")

        """Create geometries for negative pixel classes"""

        positive_labels['eebbox'] = positive_labels.apply(lambda x: bounding_box_from_center(x.latitude, x.longitude, 1000), axis = 1)

        self.positive_ee_geos = []
        for _, df in positive_labels.groupby(['began', 'ended']):
           self.positive_ee_geos.append(
               bounding_boxes_to_polygon(df['eebbox']).buffer(1000)
           )
        self.negative_ee_geos = [self.ee_region_geo.difference(geo) for geo in self.positive_ee_geos]

        """Use geometries to sample negative pixels"""

        perm_water_mask = lambda img: img.select('jrc_perm_water').gt(0).And(img.select('flooded').eq(0))
        non_flood_mask = lambda img: img.select('flooded').eq(0).And(img.select('jrc_perm_water').eq(0))

        logger.info("Sampling permanent water bodies.")

        perm_water = (c2sdfo
            .filterBounds(self.ee_region_geo)
            .filterDate(ee.Date(start_date), ee.Date(end_date))
            .map(lambda img: img.addBands(img.pixelLonLat()))
            .map(lambda img: img.updateMask(perm_water_mask(img)))
        )

        perm_water_df = self.dicts_to_dataframe(
                                    ee.List(
                                        perm_water
                                        .iterate(
                                            self.sample_img_to_data_dict, 
                                            first=ee.List([ee.String('complete'), ee.Number(0), ee.List([])])
                                        )
                                    )
                                    .get(2)
                                    .getInfo()
                                )

        """There may not be enough permanent water samples to take, so we can add the deficit to the other negative samples"""

        perm_water_df.dropna(inplace=True)

        self.n_neg_per_img = (
            np.array(self.n_neg_per_img) + 
            (np.array(self.n_neg_per_img) - 3 * perm_water_df.groupby(['began', 'ended']).size().values)
        ).tolist()


        logger.info("Sampling negative pixels near positive pixels.")

        negative_floods_near_positive = (c2sdfo
            .filterBounds(self.ee_region_geo)
            .filterDate(ee.Date(start_date), ee.Date(end_date))
            .map(lambda img: img.addBands(img.pixelLonLat()))
            .map(lambda img: img.updateMask(non_flood_mask(img)))
        )

        negative_floods_near_positive_df = self.dicts_to_dataframe(
                                                ee.List(
                                                    negative_floods_near_positive
                                                    .iterate(
                                                        self.sample_img_to_data_dict, 
                                                        first=ee.List([ee.String('positive'), ee.Number(0), ee.List([])])
                                                    )
                                                )
                                                .get(2)
                                                .getInfo()
                                            )

        logger.info("Sampling negative pixels from areas not near positive pixels.")

        negative_floods_not_near_positive = (c2sdfo
            .filterBounds(self.ee_region_geo)
            .filterDate(ee.Date(start_date), ee.Date(end_date))
            .map(lambda img: img.addBands(img.pixelLonLat()))
            .map(lambda img: img.updateMask(non_flood_mask(img)))
        )

        negative_floods_not_near_positive_df = self.dicts_to_dataframe(
                                                    ee.List(
                                                        negative_floods_not_near_positive
                                                        .iterate(
                                                            self.sample_img_to_data_dict, 
                                                            first=ee.List([ee.String('negative'), ee.Number(0), ee.List([])])
                                                        )
                                                    )
                                                    .get(2)
                                                    .getInfo()
                                                )

        negative_samples_df = pd.concat([
            perm_water_df,
            negative_floods_near_positive_df,
            negative_floods_not_near_positive_df
        ])

        """Clean up the unpassed variables"""
        del (
            perm_water_df, 
            negative_floods_near_positive_df,
            negative_floods_not_near_positive_df,
        )

        return negative_samples_df

    
    def reduce_img_to_data_dict(self, img: ee.Image, prev: ee.List=ee.List([])) -> ee.List:
        """
        Function to be used with Earth Engine's .iterate() to turn all unmasked pixels of
        an Image in an ImageCollection into a data dictionary.
        """
        data_dict = (
            img
            .reduceRegion(
                ee.Reducer.toList(),
                self.ee_region_geo,
                maxPixels=200000000
            )
        )

        n = ee.List(data_dict.get(data_dict.keys().get(0))).length()

        date_dict = ee.Dictionary({
            'began': ee.List.repeat(img.get('began'), n),
            'ended': ee.List.repeat(img.get('ended'), n)
        })

        data_dict = data_dict.combine(date_dict)

        return ee.List(prev).add(data_dict)

    def sample_img_to_data_dict(
            self, 
            img: ee.Image, 
            prev: ee.List=ee.List([ee.String('positive'), ee.Number(0), ee.List([])])
        ) -> ee.List:
        """
        Function to be used with Earth Engine's .iterate() to sample from unmasked pixels of
        an Image in an ImageCollection and turn them into a data dictionary.

        To be used when region of interest is too large to reduce entirely. The tuple `prev` helps
        determine which geometries to use for sampling and ensures that the number of samples taken
        from each image is distributed in the same way as the positive labels:
            - 'positive': self.positive_ee_geos
            - 'negative': self.negative_ee_geos
            - 'complete': self.ee_region_geo
        """
        prev = ee.List(prev)
        geo_str, img_pos, prev = ee.String(prev.get(0)), ee.Number(prev.get(1)), ee.List(prev.get(2))

        region = ee.Algorithms.If(
            geo_str.equals('positive'),
            ee.List(self.positive_ee_geos).get(img_pos),
            ee.Algorithms.If(
                geo_str.equals('negative'),
                ee.List(self.negative_ee_geos).get(img_pos),
                self.ee_region_geo
            )
        )

        n_sample = ee.Number(ee.List(self.n_neg_per_img).get(img_pos)).divide(ee.Number(3)).round()

        features = (
                    ee.FeatureCollection(
                        img
                        .sample(
                            region=region,
                            numPixels=100000,
                            geometries=False,
                            seed=4
                        )
                    )
        ).limit(n_sample)
        
        data_dict = (
            ee.Dictionary(
                features
                .iterate(
                    self.append_dicts,
                    first=features.first().toDictionary().map(lambda k, v: [])
                )
            )
        )

        date_dict = ee.Dictionary({
            'began': ee.List.repeat(img.get('began'), n_sample),
            'ended': ee.List.repeat(img.get('ended'), n_sample)
        })

        data_dict = data_dict.combine(date_dict)

        img_pos = img_pos.add(1)

        return ee.List([ee.String(geo_str), ee.Number(img_pos), ee.List(prev).add(data_dict)])

    def append_dicts(self, feat: ee.Feature, prev: ee.Dictionary) -> ee.Dictionary:
        """
        Maps over a list of Dictionaries with equivalent keys 
        to create one large Dictionary
        """
        feat_dict = ee.Feature(feat).toDictionary()
        prev = ee.Dictionary(prev)
        new_dict = prev.map(
                    lambda key, value:
                        ee.List(value).cat(ee.List([feat_dict.get(key)]))
                    )
        return new_dict
    
    def dicts_to_dataframe(self, data_dicts: list[dict]) -> pd.DataFrame:
        """
        Takes a list of dictionaries with uneven key-value pair lengths produced from
        `img_to_data_dict` and returns a pandas DataFrame
        """
        for d in data_dicts:
            max_len = np.max([len(x) for x in d.values()])
            for k, v in d.items():
                if len(v) < max_len:
                    d[k] += [np.nan] * (max_len - len(v))
        
        dfs = [pd.DataFrame(d) for d in data_dicts]

        return pd.concat(dfs)
        
    def export(
        self, 
        start_date: date, 
        end_date: date, 
        n_positive_labels: int=1000,
        negative_to_positive_ratio: float=1.0
    ) -> None:
        """
        Download the Cloud-to-Street/DFO labels and store in the output folder

        Args:
            - start_date: start date of data export window
            - end_date: end date of data export window
            - n_labels: total number of labels to export 
                (currently script creates balanced set of positive and negative labels)
            
        """

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        logger.info("Beginning export of Cloud-to-Street/DFO flood labels")
        logger.info("Finding positive labels")
        positive_labels = self.find_positive_labels(start_date, end_date, n_positive_labels)

        logger.info("Positive labels found; they look like this:")
        logger.info(positive_labels.head())

        logger.info(f"{positive_labels.groupby(['began', 'ended']).ngroups} floods occurred over this region during the study period.")

        logger.info("Finding negative labels")
        negative_labels = self.find_negative_labels(positive_labels.copy(), start_date, end_date, negative_to_positive_ratio, logger)

        logger.info("Negative labels found; they look like this:")
        logger.info(negative_labels.head())

        flood_labels = pd.concat([positive_labels, negative_labels], axis = 0)

        if self.region_type == 'multiple':
            region_name = "_".join(self.region).lower()
        else:
            region_name = self.region.lower()

        outpath = os.path.join(
            self.output_folder, 
            f"flood_labels_{region_name}.csv"
        )

        logger.info(f"Writing flood labels to {outpath}")
        flood_labels.to_csv(outpath, index=False)