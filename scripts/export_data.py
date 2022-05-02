import sys
import logging
import argparse
from pathlib import Path
from typing import Union

sys.path.append("..")

from src.exporters import (
    CropHarvestSentinel2Exporter,
    C2SDFOSentinel1Exporter
)
from src.utils.regions import STR2BB, REGIONS, _check_region

base_data_dir = Path(__file__).resolve().parents[1] / 'data'

def export_cropharvest_sentinel2(data_dir: Path, region: Union[str, list[str]], combine_regions: bool=False) -> None:
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

def export_c2sdfo_sentinel1(data_dir: Path, region: Union[str, list[str]], combine_regions: bool=False) -> None:
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

    _check_region(args.region)

    data_dir = (
        (base_data_dir / args.region.lower()) if isinstance(args.region, str) else (base_data_dir / "_".join(args.region).lower())
    )

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    export_cropharvest_sentinel2(data_dir, args.region, args.combine_regions)
    export_c2sdfo_sentinel1(data_dir, args.region, args.combine_regions)