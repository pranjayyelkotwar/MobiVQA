#!/usr/bin/env python3
# coding: utf-8
"""
This script performs evaluation for VQA using early exit strategies.
It configures the data modules and model based on the provided Hydra config,
loads the model onto the appropriate device, and runs evaluation.
"""

import hydra
import pytorch_lightning as pl
import torch

from vqa import logger
from vqa.train import do_ee_eval
from vqa.train.data import VqaCrossEarlyExitDataModule
from vqa.train.data import VqaEarlyExitDataModule
from vqa.train.vqa import VqaCrossEarlyExitModule
from vqa.train.vqa import VqaEarlyExitModule


@hydra.main(config_path="conf", config_name="early_exit")
def my_app(cfg) -> None:
    """
    Main application entry point.
    
    Configures and initializes the data module and model based on the config.
    Sets up the model for evaluation and runs the evaluation procedure.
    """
    print("Starting evaluation process...")
    logger.info("Configuration:")
    logger.info(cfg)
    
    print("Seeding random generators for reproducibility...")
    pl.seed_everything(1)
    
    if cfg.use_cross_exit:
        print("Using cross exit modules.")
        dm = VqaCrossEarlyExitDataModule(cfg)
        pl_model = VqaCrossEarlyExitModule(cfg)
    else:
        print("Using standard early exit modules.")
        dm = VqaEarlyExitDataModule(cfg)
        pl_model = VqaEarlyExitModule(cfg)
    
    print("Setting model to evaluation mode and freezing parameters...")
    pl_model.eval()
    pl_model.freeze()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to device: {device}")
    pl_model.to(device)
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Number of GPUs available: {num_gpus}")
    logger.info(f'Using {num_gpus} GPUs')
    
    print("Wrapping model with DataParallel for multi-GPU support (if applicable)...")
    pl_model = torch.nn.DataParallel(pl_model)
    
    print("Starting evaluation of the model...")
    do_ee_eval(dm, pl_model, cfg.out_file)
    print("Evaluation complete. Results saved to:", cfg.out_file)
    logger.info("All done")


if __name__ == "__main__":
    print("Launching the evaluation script...")
    my_app()
