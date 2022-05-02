from pathlib import Path
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

def train_model(model: pl.LightningModule, hparams: Namespace) -> pl.LightningModule:

    hparams.data_folder = Path(hparams.data_folder)
    hparams.project_dir = hparams.data_folder.resolve().parents[1]

    callbacks = []

    if hparams.checkpoint:
        model_checkpoint_dir = hparams.project_dir / 'models' / hparams.model_type
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
        log_model=hparams.checkpoint
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

