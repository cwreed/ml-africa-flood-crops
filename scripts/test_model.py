from pathlib import Path
import sys
from argparse import ArgumentParser
from shutil import copyfile

import wandb

sys.path.append('..')

from src.models import test_model

project_dir = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model_type', type=str, help='One of [cropland, flood]')
    parser.add_argument('--download', dest='download', action='store_true', help='Flag to download best model from W&B')

    args = parser.parse_known_args()[0]

    if args.model_type == 'cropland':
        model_dir = project_dir / 'models' / 'cropland'
    elif args.model_type == 'flood':
        model_dir = project_dir / 'models' / 'flood'
    else:
        raise ValueError("model_type must be one of [cropland, flood]")

    model_dir.mkdir(exist_ok=True, parents=True)

    if args.download:
        artifact_name = (
            'cwreed/ml-africa-flood-crops/model-2ilwx0ef:v0' 
            if 
                args.model_type == 'cropland' 
            else 
                'cwreed/ml-africa-flood-crops/model-2a62r3s8:v0'
        )
        run = wandb.init()
        artifact = run.use_artifact(artifact_name, type='model')
        artifact_dir = artifact.download()

        copyfile(f"{artifact_dir}/model.ckpt", model_dir / 'model.pth')
    
    model_path = model_dir / 'model.pth'

    assert model_path.exists(), f"Cannot find model at {model_path}"

    test_model(model_path, project_dir / 'data/nigeria')