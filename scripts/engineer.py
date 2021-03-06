import sys
import argparse
import logging
from pathlib import Path
from typing import Union

sys.path.append('..')

from src.engineers import (
    CropHarvestEngineer,
    C2SDFOEngineer
)
from src.utils.regions import STR2BB, REGIONS, _check_region

base_data_dir = Path(__file__).resolve().parents[1] / 'data'

def engineer_cropharvest(
    data_dir: Path,
    region: Union[str, list[str]], 
    combine_regions: bool=False,
    val_set_size: float=0.1,
    test_set_size: float=0.2
) -> None:

    engineer = CropHarvestEngineer(data_dir, region=region, combine_regions=combine_regions)
    engineer.engineer(
        val_set_size=val_set_size,
        test_set_size=test_set_size,
        nan_fill=0.0,
        max_nan_ratio=0.3,
        checkpoint=True,
        calculate_normalizing_dict=True,
        days_per_timestep=30,
        sliding_window=False
    )

def engineer_c2sdfo(
    data_dir: Path,
    region: Union[str, list[str]], 
    combine_regions: bool=False,
    val_set_size: float=0.1,
    test_set_size: float=0.2
) -> None:

    engineer = C2SDFOEngineer(data_dir, region=region, combine_regions=combine_regions)
    engineer.engineer(
        val_set_size=val_set_size,
        test_set_size=test_set_size,
        nan_fill=0.0,
        max_nan_ratio=0.3,
        checkpoint=True,
        calculate_normalizing_dict=True,
        days_per_timestep=12,
        sliding_window=True,
        n_timesteps_per_instance=6
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Data engineer script")
    parser.add_argument("--region", help=f"One of {REGIONS.keys()} or one or more of {STR2BB.keys()}. Should match what was passed to `export_labels.py`.")
    parser.add_argument("--combine_regions", type=bool, default=False, help="Whether or not to combine regions into one. Should match what was passed to `export_labels.py`.")
    parser.add_argument("--val_set_size", type=float, default=0.1, help="Value between [0,1] corresponding to the desired size of the validation set for each dataset.")
    parser.add_argument("--test_set_size", type=float, default=0.2, help="Value between [0,1] corresponding to the desired size of the test set for each dataset.")

    args = parser.parse_args()

    _check_region(args.region)

    data_dir = (
        (base_data_dir / args.region.lower()) if isinstance(args.region, str) else (base_data_dir / "_".join(args.region).lower())
    )

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    engineer_cropharvest(data_dir, args.region, args.combine_regions, args.val_set_size, args.test_set_size)
    engineer_c2sdfo(data_dir, args.region, args.combine_regions, args.val_set_size, args.test_set_size)
