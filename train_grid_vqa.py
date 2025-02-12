#!/usr/bin/env python3
# coding: utf-8

# Description:
# This script serves as the entry point for training a VQA model using PyTorch Lightning.
# It loads the configuration via Hydra, initializes the data and model modules, sets up checkpointing,
# creates the Trainer, and starts the training process.
# This file can be referenced in the main README for an overview of the training pipeline.

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf

from vqa import logger
from vqa.train.data import VqaDataModule
from vqa.train.io import CheckpointEveryNSteps
from vqa.train.vqa import VqaLightningModule


@hydra.main(config_path="conf", config_name="finetune_grid")
def my_app(cfg) -> None:
    # Log the configuration for debugging
    logger.info("Loaded configuration:")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    print("Configuration loaded and printed.")

    # Set seed for reproducibility
    pl.seed_everything(1)
    print("Random seed set to 1.")

    # Initialize Data Module
    print("Initializing data module...")
    dm = VqaDataModule(cfg)
    print("Data module initialized.")

    # Initialize Model Module
    print("Initializing model...")
    model = VqaLightningModule(cfg)
    print("Model initialized.")

    # Set up checkpoint callbacks: top-k based on validation accuracy and custom step checkpointing
    print("Setting up checkpoint callbacks...")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=cfg.weights_save_path,
        save_weights_only=cfg.save_weights_only,
        filename="grid-vqa-{epoch:02d}-{val_acc:.4f}",
        save_top_k=3,
        mode="max",
    )
    step_checkpoint = CheckpointEveryNSteps(
        cfg.save_step_frequency,
        cfg.save_step_max_keep,
    )
    print("Checkpoint callbacks set.")

    # Initialize Trainer with given configuration
    print("Initializing trainer...")
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, step_checkpoint],
        gpus=cfg.num_gpus,
        precision=cfg.hparams.precision,
        amp_level=cfg.hparams.amp_level,
        stochastic_weight_avg=cfg.hparams.stochastic_weight_avg,
        accelerator="dp",
        deterministic=True,
        log_every_n_steps=cfg.log_every_n_steps,
        max_epochs=cfg.hparams.max_epochs,
        overfit_batches=cfg.overfit_batches,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        gradient_clip_val=cfg.hparams.gradient_clip_val,
    )
    print("Trainer initialized.")

    # Begin training
    print("Starting training...")
    trainer.fit(model, datamodule=dm)
    print("Training ended.")

    # Final log message
    logger.info("All done")
    print("All done.")


if __name__ == "__main__":
    print("Launching the VQA training application...")
    my_app()
    print("Application finished.")