#!/usr/bin/env python3
# coding: utf-8

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
from transformers import AutoConfig
from transformers import LxmertTokenizer

from vqa import logger
from vqa.lxmert.modeling_lxmert import LxmertForQuestionAnswering
from vqa.utils import Timer
from vqa.utils import timings
from vqa.vision.processing_image import ImageLoader
from vqa.vision.roi_heads import add_attribute_config


def setup(cfg_file, model_weights=None):
    """
    Create configurations and perform basic setups.
    Loads configuration file, set model weights, and device.
    """
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(cfg_file)
    # Force the final residual block to use dilation 1.
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.MODEL.WEIGHTS = model_weights
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    default_setup(cfg, None)
    print("[setup] Configuration loaded and setup complete.")
    return cfg


@hydra.main(config_path="conf", config_name="eval_grid")
def my_app(cfg) -> None:
    # Log and print basic configuration and process info.
    logger.info("Loaded configuration:")
    logger.info(cfg)
    print(f"[INFO] Process ID: {os.getpid()}")

    # Load mapping of label ids to answers.
    label2answer_file = cfg.label2answer_file
    vqa_answers = json.load(open(label2answer_file))
    model_cfg = cfg.model_cfg

    # Setup Faster R-CNN part of the model.
    print("[INFO] Setting up Faster R-CNN model...")
    frcnn_cfg = setup(model_cfg, cfg.model_weights)
    frcnn = build_model(frcnn_cfg)
    frcnn.eval()
    DetectionCheckpointer(frcnn, save_dir=frcnn_cfg.OUTPUT_DIR).resume_or_load(
        frcnn_cfg.MODEL.WEIGHTS, resume=True
    )

    device = torch.device(frcnn_cfg.MODEL.DEVICE)
    print(f"[INFO] Device in use: {device}")

    # Initialize image loader.
    imag_loader = ImageLoader(frcnn_cfg)

    # Setup LXMERT tokenizer and model for question answering.
    print("[INFO] Loading LXMERT tokenizer and model...")
    lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    config = AutoConfig.from_pretrained("unc-nlp/lxmert-vqa-uncased")
    config.use_x_layers = cfg.use_x_layers
    lxmert_vqa = LxmertForQuestionAnswering(config)
    if cfg.vqa_model_weights:
        logger.info(f"loading model states from {cfg.vqa_model_weights}")
        print(f"[INFO] Loading model states from {cfg.vqa_model_weights}")
        model_states = torch.load(cfg.vqa_model_weights, map_location="cpu")
        if 'state_dict' in model_states:
            model_states = model_states['state_dict']
        for key in list(model_states.keys()):
            if "vqa_model." in key:
                logger.info(f"replacing {key}...")
                model_states[key.replace("vqa_model.", "")] = model_states[key]
                del model_states[key]
        lxmert_vqa.load_state_dict(model_states)
    lxmert_vqa.to(device)

    # Prepare source for image data and question-answer data.
    img_folder = cfg.img_folder
    qa_data = [json.loads(line) for line in open(cfg.vqa_data_file)]
    if cfg.profile_num:
        qa_data = qa_data[: cfg.profile_num]
    total = len(qa_data)

    correct = 0
    count = 0
    val_data = []

    # Start measuring total time.
    start_time = time.perf_counter()
    prof_info = 'profiling mode' if os.environ.get('profile', False) else ''
    logger.info(f"start inference...{prof_info}")
    print("[INFO] Beginning inference loop...")
    num_grids = 0  # 0 means dynamic setting based on num_grids_ratio.
    e2e_latency = []

    # Iterate over each question-answer item.
    for qa_item in qa_data:
        iter_start_time = time.perf_counter()
        img_id = qa_item["img_id"]
        q_id = qa_item["question_id"]
        question = qa_item["sent"]
        label = qa_item.get("label", None)

        # Create image file path.
        img_file = os.path.join(img_folder, img_id + ".jpg")
        print(f"[DEBUG] Processing image: {img_file} with question id: {q_id}")
        
        with Timer("img_prep"):
            # Load and preprocess the image.
            images = imag_loader(img_file)
        with Timer("img_cnn"), torch.no_grad():
            # Extract CNN features.
            feat = frcnn.backbone(images.tensor)["res5"]
        b, c, h, w = feat.shape
        img_max_features = cfg.img_max_features
        if img_max_features:
            # Flatten spatial dimensions and pad if necessary.
            feat = feat.view(b, c, -1)
            pad_feat = torch.zeros(
                (b, c, img_max_features), device=feat.device, dtype=torch.float
            )
            pad_feat[:, :, : h * w] = feat
            img_feat = pad_feat.permute(0, 2, 1)
        else:
            img_feat = feat

        # Dynamically adjust the number of grids based on ratio.
        num_grids_ratio = cfg.get("num_grids_ratio", 1)
        assert 0 < num_grids_ratio <= 1, f"{num_grids_ratio} must be (0,1]"
        num_grids = int(h * w * num_grids_ratio)
        img_feat = img_feat[:, :num_grids, :]

        # Tokenize the question.
        with Timer("q_tok"):
            inputs = lxmert_tokenizer(
                question,
                padding="max_length",
                max_length=20,
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.pin_memory().to(device, non_blocking=True)
            attention_mask = inputs.attention_mask.pin_memory().to(device, non_blocking=True)
            token_type_ids = inputs.token_type_ids.pin_memory().to(device, non_blocking=True)

        # Run the LXMERT VQA model.
        with Timer("lxmert_vqa"):
            output_vqa = lxmert_vqa(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_feats=img_feat,
                visual_pos=None,
                token_type_ids=token_type_ids,
                output_attentions=False,
            )
        # Get the prediction by selecting the score with highest probability.
        with Timer("argmax"):
            pred_vqa = output_vqa["question_answering_score"][-1].argmax(-1)
        with Timer("output_cpu"):
            pred_idx = pred_vqa.cpu()
        with Timer("numpy"):
            pred = vqa_answers[pred_idx.numpy()[0]]

        # Record end-to-end latency.
        e2e_latency.append((time.perf_counter() - iter_start_time) * 1e3)
        
        # Log prediction details.
        if cfg.print_pred:
            logger.info(f"q{q_id}: {question}\n\t pred: {pred}\n\tlabel: {label}\n")
            print(f"[PREDICTION] Question ID: {q_id} | Question: {question}")
            print(f"[PREDICTION] Predicted Answer: {pred} | Label Info: {label}")
        
        # Store validation data.
        val_data.append({"question_id": int(q_id), "answer": pred})
        correct += label.get(pred, 0) if label else 0

        # Clean up to avoid GPU memory buildup.
        del feat, output_vqa, pred_vqa
        torch.cuda.empty_cache()
        count += 1

        # Clear timings on warmup phase.
        if cfg.profile_num and count == 3:
            timings.clear()
        
        # Print progress every 10 iterations.
        if count % 10 == 0:
            duration = time.perf_counter() - start_time
            avg_acc = correct / count * 100
            logger.info(
                f"{duration:.3f}s processed: {count}/{total}, acc={avg_acc:.2f}"
            )
            print(f"[INFO] {duration:.3f}s processed: {count}/{total}, Accuracy: {avg_acc:.2f}%")
    
    # Log overall accuracy.
    overall_acc = correct / count * 100
    logger.info(f"Overall accuracy: {overall_acc:.2f}%")
    print(f"[RESULT] Final Accuracy: {overall_acc:.2f}%")

    hostname = socket.gethostname()
    xl = cfg.use_x_layers or 5
    ng = num_grids
    time_str = [f"{hostname}-x{xl}-ng{ng}, num, key, avg, std, min, max"]
    for tk, tv in timings.items():
        time_str.append(
            f"{hostname}-x{xl}-ng{ng}, {len(tv)}, {tk}, "
            f"{np.mean(tv) * 1e3:.3f}, {np.std(tv) * 1e3:.3f}, "
            f"{np.min(tv) * 1e3:.3f}, {np.max(tv) * 1e3:.3f}"
        )
    logger.info("\n".join(time_str))
    logger.info(
        f"e2e-{hostname}-x{xl}-ng{ng}, {len(e2e_latency)}, "
        f"{np.mean(e2e_latency):.3f}, {np.std(e2e_latency):.3f}, "
        f"{np.min(e2e_latency):.3f}, {np.max(e2e_latency):.3f}"
    )
    print("[INFO] Inference timers:")
    for line in time_str:
        print(f"  {line}")

    # Save timing data to output file.
    with open(cfg.out_file, "w") as outfile:
        json.dump(timings, outfile)
    print(f"[INFO] Timing results written to {cfg.out_file}")

    logger.info("all done")
    print("[INFO] Processing completed successfully.")


if __name__ == "__main__":
    my_app()
