from pathlib import Path
from argparse import Namespace
import sys
from typing import Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

sys.path.append("..")

from src.models import CroplandMapper, FloodMapper

def train_model(model: Union[CroplandMapper, FloodMapper], hparams: Namespace) -> Union[CroplandMapper, FloodMapper]:

    hparams.data_folder = Path(hparams.data_folder)
    hparams.project_dir = hparams.data_folder.resolve().parents[1]

    callbacks = []

    if hparams.checkpoint:
        model_checkpoint_dir = hparams.project_dir / 'models' / hparams.data_folder.name / hparams.model_type
        model_checkpoint_dir.mkdir(exist_ok=True, parents=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=model_checkpoint_dir,
                filename=hparams.run_name
            )
        )

    if hparams.early_stop:
        callbacks.append( 
            EarlyStopping(
                monitor='validation_loss',
                min_delta=0.0001,
                patience=hparams.patience,
                verbose=True,
                mode='min'
            )
        )

    wandb_logger = WandbLogger(
        project='ml-africa-flood-crops',
        name=hparams.run_name,
        job_type=f'train_{hparams.model_type}',
        log_model=hparams.checkpoint,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        max_epochs=hparams.max_epochs,
        gradient_clip_val=0.5,
        auto_lr_find=hparams.auto_lr
    )

    if hparams.auto_lr:
        trainer.tune(model)

    wandb_logger.log_hyperparams(model.hparams)
    wandb_logger.watch(model)

    trainer.fit(model)

    return model

def test_model(model_path: Path, data_dir: Path):

    """Load in the model with torch"""
    checkpoint = torch.load(model_path)
    
    hparams_dict = checkpoint['hyper_parameters']
    hparams_dict['data_folder'] = data_dir
    hparams = Namespace(**hparams_dict)

    state_dict = checkpoint['state_dict']

    if 'cropland' in model_path.parents[0].name:
        model = CroplandMapper(hparams)
        model.load_state_dict(state_dict)
    elif 'flood' in model_path.parents[0].name:
        model = FloodMapper(hparams)
        model.load_state_dict(state_dict)

    trainer = pl.Trainer()

    trainer.test(model)
