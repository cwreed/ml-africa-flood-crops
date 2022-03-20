import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Union
from datetime import date, datetime
import ee

sys.path.append("..")

try:
    ee.Initialize()
except ee.ee_exception.EEException:
    ee.Authenticate()
    ee.Initialize()

from src.exporters import (
    CropHarvestExporter,
    C2SDFOExporter,
    STR2BB,
    REGIONS
)

data_dir = os.path.join(
    Path(__file__).resolve().parents[1],
    'data'
)

def export_cropharvest() -> None:
    exporter = CropHarvestExporter(data_folder=data_dir)
    exporter.export()

def export_c2sdfo(
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
    try:
        ee.Initialize()
    except ee.ee_exception.EEException:
        ee.Authenticate()
        ee.Initialize()

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

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    export_cropharvest()
    export_c2sdfo(args.region, start_date, end_date, args.n_positive_flood_labels, args.negative_to_positive_flood_ratio, args.combine_regions)

    


