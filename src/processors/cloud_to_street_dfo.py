import os
from pathlib import Path
from typing import Union

from .base import BaseProcessor
from src.utils.regions import combine_bounding_boxes, STR2BB, REGIONS

import numpy as np
import pandas as pd

class C2SDFOProcessor(BaseProcessor):

    dataset = 'cloud-to-street-dfo'

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
        
    def load_raw_labels(self) -> pd.DataFrame:

        if self.region_type == 'multiple':
            self.region_name = "_".join(self.region).lower()
        else:
            self.region_name = self.region.lower()

        label_file = os.path.join(self.raw_folder, f"flood_labels_{self.region_name}.csv")

        assert (
            os.path.exists(label_file)
        ), f"Cannot find {label_file}. You need to run `export_labels.py` for this region before processing."
        

        return pd.read_csv(label_file)

    def process(self) -> None:
        flood_labels = self.load_raw_labels()

        flood_labels.rename(columns={
            'longitude': 'lon',
            'latitude': 'lat'
        }, inplace=True)

        flood_labels = flood_labels.loc[:, ['lon', 'lat', 'flooded', 'jrc_perm_water', 'began', 'ended']]

        output_xr = (
            flood_labels
            .reset_index(drop=True)
            .set_index(['lon', 'lat', 'began', 'ended'])
            .to_xarray()
        )

        output_xr.to_netcdf(os.path.join(self.output_folder, f"flood_labels_{self.region_name}.nc"))



