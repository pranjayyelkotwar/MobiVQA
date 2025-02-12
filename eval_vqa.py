#!/usr/bin/env python3
# coding: utf-8
'''
The script is designed for Visual Question Answering (VQA) using a two-step model pipeline. It first loads a Faster R-CNN model to extract visual features from images, and then it uses a LXMERT-based model to answer questions based on those features. The initial part of the script sets up logging and loads necessary configurations, models, tokenizers, and datasets. Logging and print statements provide progress updates during model loading and inference.

A key function in the script is the one that extracts image features, which is implemented in the function get_img_feat(). This function constructs the image file path by combining a root directory, the image folder, and the image identifier. It then preprocesses the image using the Preprocess class before passing it through the Faster R-CNN model. The output dictionary from the model contains normalized bounding boxes and ROI features, which will be later used by the VQA model.

The Preprocess class is responsible for resizing, normalizing, and padding images. It uses a resizing utility to adjust images to a standardized size, then normalizes the image tensor based on the model's pixel mean and standard deviation. The padding step ensures that all images match a common size, which facilitates batching during inference. These preprocessed images, along with metadata about their sizes and scaling factors, are essential for the correct functioning of the Faster R-CNN model.

Within the main inference loop, the script iterates over QA items grouped by image. For each image, the visual features are extracted using get_img_feat(), and the corresponding questions are tokenized using the LXMERT tokenizer. The tokenized questions and visual features are then fed into the VQA model to generate predicted answers. The script also calculates the accuracy of predictions by comparing model outputs against the ground-truth labels, and it periodically logs processing progress, including the number of questions processed and the current accuracy. Finally, the results are saved to a JSON file for later review.

'''
import json
import logging
import os
import time
from collections import defaultdict

import torch
from transformers import LxmertForQuestionAnswering, LxmertTokenizer

from vqa.vision.modeling_frcnn import GeneralizedRCNN
from vqa.vision.processing_image import Preprocess
from vqa.utils import Config

# Configure logger
logger = logging.getLogger('vqa')
logger.setLevel(logging.INFO)
fmt_str = "%(levelname)s:%(asctime)s.%(msecs)03d: %(message)s"
fmt = logging.Formatter(fmt_str, "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)


def read_data(data_file):
    """
    Read data from a file and return a list of lines.

    Parameters:
        data_file (str): Path to the file

    Returns:
        list: List of strings where each string is a line from the file.
    """
    with open(data_file, 'r') as f:
        data = f.read().split("\n")
    return data


ROOT_DIR = '/home/qqcao/work/MobiVQA/lxmert'
VQA_URL = f"{ROOT_DIR}/data/vqa/trainval_label2ans.json"

# Load VQA answers from file
vqa_answers = json.load(open(VQA_URL))

# Load models and components
logger.info("Loading Faster RCNN model and its config...")
print("Loading Faster RCNN model and its config...")
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn_cfg.model.device = 'cuda'
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
device = torch.device(frcnn_cfg.model.device)

image_preprocess = Preprocess(frcnn_cfg)

logger.info("Loading LXMERT tokenizer and VQA model...")
print("Loading LXMERT tokenizer and VQA model...")
lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")
lxmert_vqa.to(device)

# Define dataset split and image folder
split = 'minival'  # Options: minival, nominival, test, train
img_folder = 'val2014-mscoco-images'

# Load QA data from file
qa_data = json.load(open(f"{ROOT_DIR}/data/vqa/{split}.json"))
total = len(qa_data)
logger.info(f"Total number of QA items: {total}")
print(f"Total number of QA items: {total}")

# Map each image id to its corresponding questions
image_qa = defaultdict(list)
for qa_item in qa_data:
    image_qa[qa_item['img_id']].append(qa_item)


def get_img_feat(img_id):
    """
    Extract image features using Faster RCNN.

    Parameters:
        img_id (str): Image id used to locate the image file.

    Returns:
        dict: A dictionary containing normalized boxes and ROI features.
    """
    img_file = os.path.join(ROOT_DIR, 'data', img_folder, img_id + '.jpg')
    # Preprocess image and get required inputs for frcnn
    images, sizes, scales_yx = image_preprocess(img_file)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt"
    )
    return output_dict


correct = 0
count = 0
val_data = []

start_time = time.perf_counter()
logger.info('Start inference...')
print("Starting inference process...")

# Main inference loop per image.
for img_id, qa_item_list in image_qa.items():
    print(f"Processing image id: {img_id}")
    img_feat = get_img_feat(img_id)
    normalized_boxes = img_feat.get("normalized_boxes")
    features = img_feat.get("roi_features")
    q_ids = [qa_item['question_id'] for qa_item in qa_item_list]
    questions = [qa_item['sent'] for qa_item in qa_item_list]
    labels = [qa_item['label'] for qa_item in qa_item_list if 'label' in qa_item]

    # Encode questions with LXMERT tokenizer
    inputs = lxmert_tokenizer(
        questions,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)
    features = features.to(device)
    normalized_boxes = normalized_boxes.to(device)

    # Forward pass through the VQA model
    output_vqa = lxmert_vqa(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_feats=features,
        visual_pos=normalized_boxes,
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
    )
    # Get the prediction indices
    pred_vqa = output_vqa["question_answering_score"].argmax(-1)
    for q_id, q, pred_idx, label in zip(q_ids, questions, pred_vqa.tolist(), labels):
        pred = vqa_answers[pred_idx]
        val_data.append({
            'question_id': q_id,
            'question': q,
            'img_id': img_id,
            'label': label,
            'answer': pred
        })
        print(f"Question ID: {q_id}, Question: {q}")
        print(f"Predicted Answer: {pred}, Label: {label}\n")
        correct += label.get(pred, 0)
        count += 1
        if count % 100 == 0:
            duration = time.perf_counter() - start_time
            acc = correct / count * 100
            logger.info(f'{duration:.3f}s processed: {count}/{total}, acc={acc:.2f}')
            print(f"{duration:.3f}s processed: {count}/{total}, accuracy={acc:.2f}")

final_accuracy = correct / count * 100
logger.info(f"Final accuracy: {final_accuracy:.2f}%")
print(f"Final accuracy: {final_accuracy:.2f}%")

# Save inference results to file
output_file = f'data/{split}_data.json'
with open(output_file, 'w') as outfile:
    json.dump(val_data, outfile)
logger.info(f'Results saved to {output_file}')
print(f'Results saved to {output_file}')

logger.info('All done')
print("All done")
