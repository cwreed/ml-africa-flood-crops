import sys
import logging
import argparse
from pathlib import Path
from typing import Union

sys.path.append("..")

from src.exporters import (
    CropHarvestSentinel2Exporter,
    C2SDFOSentinel1Exporter,
    STR2BB,
    REGIONS
)

data_dir = Path(__file__).resolve().parents[1] / 'data'

def export_cropharvest_sentinel2(region: Union[str, list[str]], combine_regions: bool=False) -> None:
    sentinel_2_exporter = CropHarvestSentinel2Exporter(
        data_folder=data_dir,
        region=region,
        combine_regions=combine_regions
    )
    
    sentinel_2_exporter.export_for_labels(
        days_per_timestep=30,
        n_timesteps=12,
        surrounding_meters=80,
        override_label_dates=True,
        checkpoint=True,
        monitor=False
    )

def export_c2sdfo_sentinel1(region: Union[str, list[str]], combine_regions: bool=False) -> None:
    sentinel_1_exporter = C2SDFOSentinel1Exporter(
        data_folder=data_dir,
        region=region,
        combine_regions=combine_regions
    ) 

    sentinel_1_exporter.export_for_labels(
        days_per_timestep=12,
        n_timesteps=6,
        surrounding_meters=80,
        checkpoint=True,
        monitor=False
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Satellite data export script")
    parser.add_argument("--region", help=f"One of {REGIONS.keys()} or one or more of {STR2BB.keys()}.")
    parser.add_argument("--combine_regions", type=bool, default=False, help="Whether or not to combine regions into one.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    export_cropharvest_sentinel2(args.region, args.combine_regions)
    export_c2sdfo_sentinel1(args.region, args.combine_regions)