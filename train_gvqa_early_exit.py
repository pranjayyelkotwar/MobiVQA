#!/usr/bin/env python3
# coding: utf-8

'''
This code implements training and evaluation routines for Visual Question Answering (VQA) with an early exit strategy. It supports two configurations:

Early Exit Mode: Uses a streamlined early-exit module to speed up inference by deciding dynamically when to stop processing.
Cross Early Exit Mode: Incorporates a cross-modal early exit layer that leverages multi-modal features.
Key features include:

Configuration Management: Uses Hydra for flexible and dynamic configuration.
Training Framework: Built on PyTorch Lightning for concise training loops, checkpointing, and multi-GPU support.
Custom Checkpointing: Implements a custom callback to save model checkpoints every N steps, ensuring efficient use of disk space.
Inference Pipeline: Freezes the underlying VQA model and wraps the early exit model with PyTorch's DataParallel for evaluation on multiple GPUs.
This modular design allows you to experiment with different early exit methodologies to balance inference speed and predictive accuracy.
'''

import hydra
import pytorch_lightning as pl
import torch

from vqa import logger
from vqa.train import do_ee_eval
from vqa.train.data import VqaCrossEarlyExitDataModule
from vqa.train.data import VqaEarlyExitDataModule
from vqa.train.io import CheckpointEveryNSteps
from vqa.train.vqa import VqaCrossEarlyExitModule
from vqa.train.vqa import VqaEarlyExitModule

@hydra.main(config_path="conf", config_name="early_exit")
def my_app(cfg) -> None:
    # Log the configuration settings
    logger.info("Loaded configuration:")
    logger.info(cfg)
    print("Configuration loaded. See logs for details.")

    # Set random seed for reproducibility
    pl.seed_everything(1)
    print("Random seed set to 1 for reproducibility.")

    # Choose the appropriate data module and model based on the configuration flag
    if cfg.use_cross_exit:
        print("Initializing Cross Early Exit Data Module and Model...")
        dm = VqaCrossEarlyExitDataModule(cfg)
        model = VqaCrossEarlyExitModule(cfg)
        prefix = 'gvqa-cross-ee-'
    else:
        print("Initializing Early Exit Data Module and Model...")
        dm = VqaEarlyExitDataModule(cfg)
        model = VqaEarlyExitModule(cfg)
        prefix = 'gvqa-ee-'

    # Set up checkpointing to save the best models
    print("Setting up checkpoint callbacks...")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=cfg.weights_save_path,
        save_weights_only=cfg.save_weights_only,
        filename=prefix + "{epoch:02d}-{val_acc:.4f}",
        save_top_k=3,
        mode="max",
    )

    # Create a Trainer instance with specified callbacks and training parameters
    print("Initializing the Trainer...")
    trainer = pl.Trainer(
        callbacks=[
            checkpoint_callback,
            CheckpointEveryNSteps(
                cfg.save_step_frequency,
                cfg.save_step_max_keep,
            ),
        ],
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

    # Start model training
    print("Starting training...")
    trainer.fit(model, datamodule=dm)
    print("Training completed.")

    # Set model to evaluation mode and move to the appropriate device
    model.eval()
    model.freeze()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to {device}.")
    logger.info(f'Using {torch.cuda.device_count()} GPUs')

    # Wrap model for multi-GPU evaluation
    model = torch.nn.DataParallel(model)
    print("Starting evaluation...")
    do_ee_eval(dm, model, cfg.out_file)
    print("Evaluation completed. All done.")

if __name__ == "__main__":
    print("Starting the main application...")
    my_app()
