#!/usr/bin/env python3
# coding: utf-8
"""
This module runs GVQA model predictions across multiple cross-encoder layers.
It loads the model and test data, iterates through the prediction data loader, and writes
the predictions (and optionally intermediate features) to a JSON file. Additionally, it computes
evaluation metrics if ground-truth data is provided.

Usage:
    Run the script using:
        ./predict_gvqa.py
    The configuration is handled by Hydra.
"""

import json
import hydra
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm

from vqa import logger
from vqa.train.data import VqaDataModule
from vqa.train.metrics import compute_vqa_accuracy
from vqa.train.vqa import VqaLightningModule


@hydra.main(config_path="conf", config_name="finetune_grid")
def my_app(cfg) -> None:
    """
    Main prediction application.

    This function sets up the data module, the VQA LightningModule, and runs the prediction loop.
    It saves the output predictions (and optionally features) to the specified output files.
    If local evaluation is enabled, it computes and logs the accuracy using the ground-truth data.

    Args:
        cfg: Configuration object provided by Hydra.
    """
    print("Starting prediction process...")
    logger.info(cfg)
    pl.seed_everything(1)

    # Setup data module and model
    dm = VqaDataModule(cfg)
    print("Initializing model...")
    pl_model = VqaLightningModule(cfg)
    pl_model.eval()
    pl_model.freeze()

    # Handle multi-GPU if available
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pl_model.to(device)
        logger.info(f'Using {torch.cuda.device_count()} GPUs')
        print(f"Model moved to device: {device} and wrapped for DataParallel.")
        pl_model = torch.nn.DataParallel(pl_model)

    print("Setting up prediction data...")
    dm.setup('predict')
    label2ans = json.load(open(cfg.label2ans_file))
    use_x_layers = cfg.use_x_layers
    predictions = [[] for _ in range(use_x_layers)]
    out_dir = Path(cfg.out_file).parent
    feat_dir = out_dir / cfg.save_feature_path
    feat_dir.mkdir(exist_ok=True)
    print("Starting the prediction loop...")

    # Process prediction batches
    for batch in tqdm(dm.predict_dataloader(), desc='Predicting...'):
        output_vqa = pl_model(batch)
        q_ids = output_vqa['qid']
        qa_scores = output_vqa["qa_scores"]
        cross_attn = output_vqa["cross_encoder_attentions"]
        for layer, i_pred in enumerate(qa_scores):
            preds = i_pred.argmax(-1)
            layer_attn = cross_attn[layer]
            for i, qid in enumerate(q_ids):
                pred_idx = preds[i]
                pred_ans = label2ans[pred_idx]
                predictions[layer].append({
                    "question_id": int(qid),
                    "pred": int(pred_idx),
                    "answer": pred_ans,
                })
                if cfg.save_feature_path:
                    logits = i_pred[i]
                    ex_attn = layer_attn[i]
                    feat_file = feat_dir / f'{int(qid)}.th'
                    with open(feat_file, "wb") as f:
                        # Save CPU tensors for portability
                        save = logits.cpu(), ex_attn.cpu()
                        torch.save(save, f)
                        print(f"Saved feature file for question {qid} at {feat_file}")

    # Write predictions to the output file
    with open(cfg.out_file, "w") as f:
        json.dump(predictions, f)
    print(f"Predictions saved to {cfg.out_file}")

    # Compute evaluation metrics if ground truth is available
    if cfg.local_eval:
        print("Starting local evaluation...")
        gt_file = cfg.test_file
        gt_data = [json.loads(line) for line in open(gt_file)]
        for layer, layer_predictions in enumerate(predictions):
            accuracy = compute_vqa_accuracy(gt_data, layer_predictions)
            logger.info(f"Layer {layer}: {accuracy=:.2f}")
            print(f"Layer {layer}: Accuracy = {accuracy:.2f}")
    print("All done.")
    logger.info("All done")


if __name__ == "__main__":
    my_app()
