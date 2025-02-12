#!/usr/bin/env python3
# coding: utf-8
import argparse
import json

from vqa.train.metrics import compute_vqa_accuracy


def main(args):
    """
    Main function to evaluate the VQA accuracy.

    This function loads prediction data from a JSON file and ground truth data from 
    a text file where each line represents a JSON object. It computes the VQA accuracy 
    by comparing the predictions with the ground truth and prints the accuracy formatted 
    to four decimal places.

    Parameters:
        args: An object with attributes:
            pred_file (str): The file path to the JSON file containing predictions.
            gt_file (str): The file path to the text file containing ground truth data,
                           where each line is a JSON-formatted string.

    """
    pred_file = args.pred_file
    gt_file = args.gt_file
    predictions = json.load(open(pred_file))
    gt_data = [json.loads(line) for line in open(gt_file)]
    accuracy = compute_vqa_accuracy(gt_data, predictions)
    print(f'{accuracy=:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_file', type=str)
    parser.add_argument('-g', '--gt_file', type=str)
    main(parser.parse_args())
