from pathlib import Path
import sys
import argparse
import re
import logging
from datetime import date, datetime, timedelta

import torch
import xarray as xr

import matplotlib.pyplot as plt

sys.path.append('..')

from src.engineers import CropHarvestEngineer, C2SDFOEngineer
from src.models import CroplandMapper, FloodMapper
from src.utils.regions import STR2BB, REGIONS, _check_region

base_data_dir = Path(__file__).resolve().parents[1] / 'data'
base_model_dir = Path(__file__).resolve().parents[1] / 'models'

def cropland_mapper(
    data_dir: Path,
    engineer: CropHarvestEngineer,
    inference_region_name: str,
    model_path: Path,
):
    logger = logging.getLogger(__name__)
    test_folder = data_dir / 'raw' / 'earth-engine-region-sentinel-2'
    test_files = test_folder.glob(f"*{inference_region_name}*.tif")

    checkpoint = torch.load(model_path)
    
    hparams_dict = checkpoint['hyper_parameters']
    hparams_dict['data_folder'] = data_dir
    hparams_dict['random_seed'] = 2022
    hparams = argparse.Namespace(**hparams_dict)

    state_dict = checkpoint['state_dict']

    model = CroplandMapper(hparams)
    model.load_state_dict(state_dict)

    for test_path in test_files:
        logger.info(f"Making cropland predictions for {test_path.name}")

        start_date, end_date = re.findall('\d{8}', test_path.name)
        start_date, end_date = datetime.strptime(start_date, '%Y%m%d').date(), datetime.strptime(end_date, '%Y%m%d').date()

        input_data = engineer.tif_to_np(
            filepath=test_path,
            start_date=start_date,
            add_ndvi=True,
            nan_fill=0,
            days_per_timestep=30,
            sliding_window=False,
            normalizing_dict=model.normalizing_dict,
        )[0]

        predictions = model.predict(input_data).squeeze()
        predictions = predictions.sortby('lat').sortby('lon')

        save_dir = data_dir / 'predictions' / model.__class__.__name__
        save_dir.mkdir(exist_ok=True)

        predictions.to_netcdf(save_dir / f"preds_{test_path.name}.nc")

def flood_mapper(
    data_dir: Path,
    engineer: C2SDFOEngineer,
    inference_region_name: str,
    model_path: Path,
    n_timesteps_per_pred: int=6,
    days_per_timestep: int=12
):
    logger = logging.getLogger(__name__)
    test_folder = data_dir / 'raw' / 'earth-engine-region-sentinel-2'
    test_files = test_folder.glob(f"*{inference_region_name}*.tif")

    checkpoint = torch.load(model_path)
    
    hparams_dict = checkpoint['hyper_parameters']
    hparams_dict['data_folder'] = data_dir
    hparams_dict['random_seed'] = 2022
    hparams = argparse.Namespace(**hparams_dict)

    state_dict = checkpoint['state_dict']

    model = FloodMapper(hparams)
    model.load_state_dict(state_dict)

    for test_path in test_files:
        logger.info(f"Making flood predictions for {test_path.name}")

        start_date, end_date = re.findall('\d{8}', test_path.name)
        start_date, end_date = datetime.strptime(start_date, '%Y%m%d').date(), datetime.strptime(end_date, '%Y%m%d').date()

        inference_dates = [start_date + timedelta(days = n_timesteps_per_pred * days_per_timestep)]
        while inference_dates[-1] < end_date:
            inference_dates.append(inference_dates[-1] + timedelta(days = n_timesteps_per_pred * days_per_timestep))

        input_data = engineer.tif_to_np(
            filepath=test_path,
            start_date=start_date,
            nan_fill=0,
            days_per_timestep=days_per_timestep,
            sliding_window=True,
            n_timesteps_per_instance=n_timesteps_per_pred,
            normalizing_dict=model.normalizing_dict,
        )

        predictions: list[xr.DataArray] = []

        for x in input_data:
            predictions.append(model.predict(x).squeeze())

        predictions_dict: dict[date, xr.DataArray] = {
            inference_date: preds for inference_date, preds in zip(inference_dates, predictions)
        }

        save_dir = data_dir / 'predictions' / model.__class__.__name__
        save_dir.mkdir(exist_ok=True)

        for inference_date, preds in predictions_dict.items():
            
            date_str = datetime.strftime(inference_date, '%Y%m%d')
            predictions = preds.sortby('lat').sortby('lon')

            predictions.to_netcdf(save_dir / f"preds_{date_str}_{test_path.name}.nc")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description="Satellite regional data export script")
    parser.add_argument("--region", help=f"One of {REGIONS.keys()} or one or more of {STR2BB.keys()}.")
    parser.add_argument("--combine_regions", type=bool, default=False, help="Whether or not to combine regions into one.")
    parser.add_argument("--region_boundary_geojson", type=str, default=None, help="Name of file to use for calculating regional boundary for data export; should match what was used in export_region.py")
    parser.add_argument("--map_cropland", dest="map_cropland", action='store_true')
    parser.add_argument("--map_flood", dest="map_flood", action='store_true')

    args = parser.parse_args()

    _check_region(args.region)

    data_dir = (
        (base_data_dir / args.region.lower()) if isinstance(args.region, str) else (base_data_dir / "_".join(args.region).lower())
    )

    inference_region_name = args.region_boundary_geojson[:-8]

    if args.map_cropland:
        crop_engineer = CropHarvestEngineer(data_dir, args.region, args.combine_regions)
        model_path = base_model_dir / 'cropland' / 'model.pth'

        assert model_path.exists(), f"Could not find model at {model_path}"

        cropland_mapper(data_dir, crop_engineer, inference_region_name, model_path)
    
    if args.map_flood:
        flood_engineer = C2SDFOEngineer(data_dir, args.region, args.combine_regions)
        model_path = base_model_dir / 'flood' / 'model.pth'

        assert model_path.exists(), f"Could not find model at {model_path}"

        flood_mapper(data_dir, flood_engineer, inference_region_name, model_path)



