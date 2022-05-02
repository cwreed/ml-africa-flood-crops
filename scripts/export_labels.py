import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Union
from datetime import date, datetime

sys.path.append("..")

from src.exporters import (
    CropHarvestExporter,
    C2SDFOExporter
)
from src.utils.regions import STR2BB, REGIONS, _check_region

base_data_dir = Path(__file__).resolve().parents[1] / 'data'

def export_cropharvest(data_dir: Path) -> None:
    exporter = CropHarvestExporter(data_folder=data_dir)
    exporter.export()

def export_c2sdfo(
        data_dir: Path,
        region: Union[str, list[str]], 
        start_date: date, 
        end_date: date, 
        n_positive_labels: int,
        negative_to_positive_ratio: float,
        combine_regions: bool=False
    ) -> None:
    exporter = C2SDFOExporter(data_folder=data_dir, region=region, combine_regions=combine_regions)
    exporter.export(start_date=start_date, end_date=end_date, n_positive_labels=n_positive_labels, negative_to_positive_ratio=negative_to_positive_ratio)


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Data export script")
    parser.add_argument("--region", help=f"One of {REGIONS.keys()} or one or more of {STR2BB.keys()}.")
    parser.add_argument("--combine_regions", type=bool, default=False, help="Whether or not to combine regions into one.")
    parser.add_argument("--start_date", type=str, help="Start date of data export window in form YYYY-MM-DD.")
    parser.add_argument("--end_date", type=str, help="End date of data export window in format YYYY-MM-DD.")
    parser.add_argument("--n_positive_flood_labels", type=int, default=1000, help="Number of positive flood labels to export.")
    parser.add_argument("--negative_to_positive_flood_ratio", type=float, default=1.0, help="Number of negative labels to export for each positive label.")

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    _check_region(args.region)

    data_dir = (
        (base_data_dir / args.region.lower()) if isinstance(args.region, str) else (base_data_dir / "_".join(args.region).lower())
    )

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    export_cropharvest(data_dir)
    export_c2sdfo(data_dir, args.region, start_date, end_date, args.n_positive_flood_labels, args.negative_to_positive_flood_ratio, args.combine_regions)

    


