from pathlib import Path
from typing import Union

from .base import BaseProcessor
from src.utils.regions import _initialize_regions

import pandas as pd

class C2SDFOProcessor(BaseProcessor):

    dataset = 'cloud-to-street-dfo'

    def __init__(self, data_folder: Path, region: Union[str, list[str]], combine_regions: bool=False):
        super().__init__(data_folder)

        _initialize_regions(self, region=region, combine_regions=combine_regions)
        
    def load_raw_labels(self) -> pd.DataFrame:

        if self.region_type == 'multiple':
            self.region_name = "_".join(self.region).lower()
        else:
            self.region_name = self.region.lower()

        label_file = self.raw_folder / f"flood_labels_{self.region_name}.csv"

        assert (
            label_file.exists()
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

        output_xr.to_netcdf((self.output_folder / f"flood_labels_{self.region_name}.nc"))



