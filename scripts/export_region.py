import sys
import logging
import argparse
from pathlib import Path
from typing import Union
from datetime import date, datetime

sys.path.append("..")

from src.exporters import (
    RegionalSentinel1Exporter,
    RegionalSentinel2Exporter
)
from src.utils.regions import STR2BB, REGIONS, _check_region

base_data_dir = Path(__file__).resolve().parents[1] / 'data'

def export_sentinel_1_region(
    data_dir: Path, 
    region: Union[str, list[str]],
    region_boundary_file_name: str,
    inference_start_date: date,
    inference_end_date: date,
    combine_regions: bool=False
):
    sentinel_1_exporter = RegionalSentinel1Exporter(
        data_folder=data_dir, region=region, combine_regions=combine_regions
    )

    sentinel_1_exporter.export_for_region(
        region_boundary_file=region_boundary_file_name,
        inference_start_date=inference_start_date,
        inference_end_date=inference_end_date,
        days_per_timestep=12,
        n_timesteps=6,
        checkpoint=True,
        monitor=False,
    )

def export_sentinel_2_region(
    data_dir: Path, 
    region: Union[str, list[str]],
    region_boundary_file_name: str,
    inference_date: date,
    combine_regions: bool=False
):
    sentinel_2_exporter = RegionalSentinel2Exporter(
        data_folder=data_dir, region=region, combine_regions=combine_regions
    )

    sentinel_2_exporter.export_for_region(
        region_boundary_file=region_boundary_file_name,
        inference_date=inference_date,
        days_per_timestep=30,
        n_timesteps=12,
        checkpoint=True,
        monitor=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite regional data export script")
    parser.add_argument("--region", help=f"One of {REGIONS.keys()} or one or more of {STR2BB.keys()}.")
    parser.add_argument("--combine_regions", type=bool, default=False, help="Whether or not to combine regions into one.")
    parser.add_argument("--crop_inference_date", type=str, help="Date of Sentinel-2 data inference window in form YYYY-MM-DD.")
    parser.add_argument("--flood_inference_start_date", type=str, help="First date of Sentinel-1 data inference window in form YYYY-MM-DD.")
    parser.add_argument("--flood_inference_end_date", type=str, help="Last date of Sentinel-1 data inference window in form YYYY-MM-DD.")
    parser.add_argument("--region_boundary_geojson", type=str, default=None, help="Name of file to use for calculating regional boundary for data export; can be None, defaulting export to `region`")

    args = parser.parse_args()

    _check_region(args.region)

    data_dir = (
        (base_data_dir / args.region.lower()) if isinstance(args.region, str) else (base_data_dir / "_".join(args.region).lower())
    )

    crop_inference_date = datetime.strptime(args.crop_inference_date, '%Y-%m-%d')
    flood_inference_start_date = datetime.strptime(args.flood_inference_start_date, '%Y-%m-%d')
    flood_inference_end_date = datetime.strptime(args.flood_inference_end_date, '%Y-%m-%d')

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    export_sentinel_2_region(
        data_dir, 
        region=args.region, 
        region_boundary_file_name=args.region_boundary_geojson,
        inference_date=crop_inference_date,
        combine_regions=args.combine_regions
    )

    export_sentinel_1_region(
        data_dir,
        region=args.region,
        region_boundary_file_name=args.region_boundary_geojson,
        inference_start_date=flood_inference_start_date,
        inference_end_date=flood_inference_end_date,
        combine_regions=args.combine_regions
    )