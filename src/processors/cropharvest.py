import os
from pathlib import Path
from typing import Union

from .base import BaseProcessor
from src.utils.regions import combine_bounding_boxes, STR2BB, REGIONS

import numpy as np
import pandas as pd
import geopandas as gpd

class CropHarvestProcessor(BaseProcessor):

    dataset = 'cropharvest'

    def __init__(self, data_folder: Path, region: Union[str, list[str]], combine_regions: bool=False):
        super().__init__(data_folder)

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
    
    def load_raw_labels(self) -> gpd.GeoDataFrame:

        label_file = os.path.join(self.raw_folder, "labels.geojson")

        assert (
            os.path.exists(label_file)
        ), f"Cannot find {label_file}. You must run `export_labels.py` before processing."
        

        return gpd.read_file(label_file, driver='GEOJSON')

    def filter_labels_to_region(self, labels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Filters the labels to include only points falling within the region(s) of interest.
        """
        if self.region_type == 'single':
            coords_mask = lambda lon, lat: (
                (lon >= self.region_bbox.min_lon) &
                (lon <= self.region_bbox.max_lon) &
                (lat >= self.region_bbox.min_lat) & 
                (lat <= self.region_bbox.max_lat)
            )

            labels_mask = labels.apply(lambda x: coords_mask(x.lon, x.lat), axis = 1)

            labels_filtered = labels[labels_mask]
        
        elif self.region_type == 'multiple':
            coords_masks = [
                lambda lon, lat: (
                    (lon >= bbox.min_lon) &
                    (lon <= bbox.max_lon) &
                    (lat >= bbox.min_lat) & 
                    (lat <= bbox.max_lat)
                )
                for bbox in self.region_bbox
            ]

            labels_masks = [labels.apply(lambda x: coords_mask(x.lon, x.lat), axis = 1) for coords_mask in coords_masks]
            labels_mask = np.logical_or.reduce(labels_masks)

            labels_filtered = labels[labels_mask]
        
        return labels_filtered
    
    def process(self) -> None:
        """
        Filters the raw CropHarvest labels to the region of interest, then transforms the labels into 
        a NetCDF file.
        """
        crop_labels = self.load_raw_labels()
        filtered_crop_labels = self.filter_labels_to_region(crop_labels)

        filtered_crop_labels['collection_date'] = pd.to_datetime(filtered_crop_labels['collection_date'])

        filtered_crop_labels = filtered_crop_labels.loc[:, ['lon', 'lat', 'is_crop', 'collection_date']]

        output_xr = (
            filtered_crop_labels
            .reset_index(drop=True)
            .set_index(['lon', 'lat', 'collection_date'])
            .to_xarray()
        )

        if self.region_type == 'multiple':
            region_name = "_".join(self.region).lower()
        else:
            region_name = self.region.lower()

        output_xr.to_netcdf(os.path.join(self.output_folder, f'crop_labels_{region_name}.nc'))
