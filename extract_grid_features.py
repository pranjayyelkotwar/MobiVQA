#!/usr/bin/env python3
# coding: utf-8
"""
This script extracts grid features from images using a detectron2 based FRCNN model.
It loads the configurations via Hydra, loads images, extracts features with the CNN backbone,
and saves the features as .pth files. Progress messages are printed to the console for tracking.
"""

import json
import os
import socket
import time

import hydra
import torch
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from vqa import logger
from vqa.utils import Timer
from vqa.vision.processing_image import ImageLoader
from vqa.vision.roi_heads import add_attribute_config


def setup(cfg_file, model_weights=None):
    """
    Create and configure the detectron2 settings.

    Args:
        cfg_file (str): Path to the model configuration file.
        model_weights (str, optional): Path to the model weights file.
    
    Returns:
        cfg (CfgNode): The frozen configuration node.
    """
    print("Setting up configuration...")
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(cfg_file)
    # force the final residual block to have dilations 1
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.MODEL.WEIGHTS = model_weights
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    default_setup(cfg, None)
    print("Configuration setup complete.")
    return cfg


@hydra.main(config_path="conf", config_name="extract_grid")
def my_app(cfg) -> None:
    """
    Main function to perform grid feature extraction.

    Steps:
      1. Log configuration details.
      2. Setup detectron2 configuration and load the FRCNN model.
      3. Load images and extract grid features.
      4. Save the extracted features.
    """
    logger.info("Loaded configuration:")
    logger.info(cfg)
    print(f"Process ID: {os.getpid()}")
    model_cfg = cfg.model_cfg

    print("Setting up the FRCNN model configuration...")
    frcnn_cfg = setup(model_cfg, cfg.model_weights)
    print("Building the FRCNN model...")
    frcnn = build_model(frcnn_cfg)
    frcnn.eval()
    DetectionCheckpointer(frcnn, save_dir=frcnn_cfg.OUTPUT_DIR).resume_or_load(
        frcnn_cfg.MODEL.WEIGHTS, resume=True
    )
    print("Model loaded and ready for feature extraction.")

    imag_loader = ImageLoader(frcnn_cfg)
    img_folder = cfg.img_folder
    os.makedirs(cfg.feature_folder, exist_ok=True)
    print(f"Image folder: {img_folder}")
    print(f"Feature folder: {cfg.feature_folder}")
    print("Loading QA data...")
    qa_data = [json.loads(line) for line in open(cfg.vqa_data_file)]
    total = len(qa_data)
    print(f"Total QA items to process: {total}")

    start_time = time.perf_counter()
    count = 0
    print("Starting feature extraction...")
    logger.info("Start extraction...")
    for qa_item in qa_data:
        img_id = qa_item["img_id"]

        # Construct the image file path. (Note: COCO dataset may need adjustment for filename)
        img_file = os.path.join(img_folder, img_id)
        print(f"Processing image: {img_file}")
        with Timer("img_prep"):
            images = imag_loader(img_file)
        with Timer("img_cnn"), torch.no_grad():
            feat = frcnn.backbone(images.tensor)["res5"]
        
        # Determine image integer id for naming the output file.
        img_int_id = int(img_id.split(".")[0].split("_")[-1])
        feature_path = os.path.join(cfg.feature_folder, f'{img_int_id}.pth')
        with open(feature_path, "wb") as f:
            torch.save(feat.cpu(), f)
        print(f"Saved features to {feature_path}")
        count += 1
        if count % 10 == 0:
            duration = time.perf_counter() - start_time
            logger.info(f"{duration:.3f}s processed: {count}/{total}")
            print(f"{duration:.3f}s: Processed {count} out of {total} images.")
    duration = time.perf_counter() - start_time
    logger.info("All done.")
    print(f"Feature extraction completed in {duration:.3f} seconds.")


if __name__ == "__main__":
    print("Starting main application...")
    my_app()
    print("Application finished.")
