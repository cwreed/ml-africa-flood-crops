import sys
import logging
import argparse
from pathlib import Path
from typing import Union

sys.path.append("..")

from src.processors import (
    CropHarvestProcessor,
    C2SDFOProcessor,
    STR2BB,
    REGIONS
)

data_dir = Path(__file__).resolve().parents[1] / 'data'

def process_cropharvest(
        region: Union[str, list[str]],
        combine_regions: bool=False
    ) -> None:

    processor = CropHarvestProcessor(data_folder=data_dir, region=region, combine_regions=combine_regions)
    processor.process()

def process_c2sdfo(
        region: Union[str, list[str]],
        combine_regions: bool=False
    ) -> None:

    processor = C2SDFOProcessor(data_folder=data_dir, region=region, combine_regions=combine_regions)
    processor.process()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Label processor script")
    parser.add_argument("--region", help=f"One of {REGIONS.keys()} or one or more of {STR2BB.keys()}. Should match what was passed to `export_labels.py`.")
    parser.add_argument("--combine_regions", type=bool, default=False, help="Whether or not to combine regions into one. Should match what was passed to `export_labels.py`.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    process_cropharvest(region=args.region, combine_regions=args.combine_regions)
    process_c2sdfo(region=args.region, combine_regions=args.combine_regions)

