'''
Below is an explanation of the code and its functions:

load_predictions(pred_file)

Opens the specified JSON file containing predictions.
Times the file loading process and prints how long the loading took.
Returns the parsed predictions.
load_gt(gt_file)

Opens a ground truth (GT) file where each line is a JSON string.
Parses each line and creates a dictionary mapping each question’s ID to its label.
The labels in the GT file are expected to be organized as key–value pairs, where keys are answer strings.
gen_pred(predictions, gt_data)

Prepares three lists (one for each prediction “layer”) to store:
The predicted label (layer_pred)
The associated probability (layer_prob)
The entropy (layer_ent)
For every prediction in each layer:
Extracts the question id, predicted index, answer string, and optionally logits.
Finds the true label by looking up the answer in the ground truth dictionary.
If logits exist, computes the probability (using a helper function calc_prob) and the entropy (using calc_entropy).
Returns the three lists – each representing the predictions, probabilities, and entropies per layer.
analyze_pred(pred_local, gt)

Uses gen_pred to obtain the per-layer predictions.
For every question from the GT:
Collects the predicted labels across layers.
Determines the “stop layer” as the first layer that gives a nonzero prediction (or the final layer if none are nonzero). This simulates an early-exit strategy.
Constructs a prediction tensor and a mapping (pred_dict) from each question id to its exit layer (adjusted to be 1-indexed).
Computes:
The average cost, using a predefined tensor of layer costs.
An overall accuracy based on the most confident answer extracted from the modified prediction tensor.
Per-layer exit accuracies.
Outputs diagnostic print statements for a subset of predictions/tensors, cost, speedup, and accuracy.
Returns the dictionary of exit layers and the modified tensor of predictions.
save_pos_gt(gt_file, gt_pos, out_file='.ee.jsonl')

Reads the original ground truth file line by line.
For each GT line, uses the provided gt_pos dictionary (which maps question IDs to a predicted exit layer) to insert a new key (ee_layer).
Writes each modified GT line as a JSON string into a new output file.
Each function is part of a pipeline that:

Loads prediction and ground truth data.
Processes predictions from multiple layers.
Decides at which layer to exit (simulate an early-exit for efficiency).
Computes metrics such as cost and speedup.
Saves the updated GT with the exit layer information.
This modular approach not only isolates different functionalities (loading, processing, analyzing, and saving) but also makes it easier to debug and extend specific parts of the workflow later on.

'''
import torch
import numpy as np
import json
import time
import copy


def load_predictions(pred_file):
    s = time.time()
    # data/vqa2-local-val-ep1-tune-all-5classifier.json
    preds_local_val = json.load(open(pred_file))
    e = time.time()
    print(f'loading took {e - s}')
    return preds_local_val
 

def load_gt(gt_file):
    # 'data/vqa/vqa2-local-val.jsonl'
    gt_data = [json.loads(line) for line in open(gt_file)]
    return {i['question_id']: i['label'] for i in gt_data}


def gen_pred(predictions, gt_data):
    layer_pred = [{} for _ in range(len(predictions))]
    layer_prob = [{} for _ in range(len(predictions))]
    layer_ent = [{} for _ in range(len(predictions))]
    
    for layer, preds in enumerate(predictions):
        for pred in preds:
            qid = pred['question_id']
            p_idx = pred['pred']
            p_ans = pred['answer']
            logits = pred.get('logits', None)
            label = gt_data[qid].get(p_ans, 0)
            prob = float(calc_prob(logits)[0, p_idx]) if logits is not None else None
            entropy = float(calc_entropy(logits)) if logits is not None else None
            layer_pred[layer][qid] = label  # (p_idx, label)
            layer_prob[layer][qid] = prob  # (p_idx, label)
            layer_ent[layer][qid] = entropy  # (p_idx, label)

    return layer_pred, layer_prob, layer_ent


def analyze_pred(pred_local, gt):
    """
    Analyze predictions generated from multiple layers against ground truth and calculate cost,
    speedup, and accuracy metrics.

    Parameters:
        pred_local (list): A list of predictions from each layer, where each element corresponds
                           to the prediction output of one layer.
        gt (dict): A dictionary representing the ground truth, where keys are question identifiers
                   and values are the corresponding correct answers.

    Returns:
        tuple: A tuple (pred_dict, new_label) containing:
            - pred_dict (dict): A dictionary mapping each question id to the 1-indexed layer number
                                 at which the prediction was determined (i.e., the layer at which the 
                                 first nonzero prediction appears, or the last layer if none are nonzero).
            - new_label (torch.Tensor): A tensor representing modified predictions; it is formed by 
                                        concatenating a zero column to the original prediction tensor 
                                        and is used in gathering the most confident answers across layers.

    Functionality:
        - Generates a new set of predictions using the gen_pred function.
        - For each question in the ground truth:
              - Extracts predictions across all layers.
              - Determines the stop layer as the first layer with a nonzero prediction (or the last layer
                if none exist), and adjusts the index to be 1-indexed.
        - Constructs a tensor of prediction indices and prints diagnostic statistics including:
              - A sample of ground truth final labels per question across layers.
              - The count and proportion of predictions per layer.
              - Computation of the average cost based on predetermined layer costs.
              - Computation of overall accuracy based on the most confident predictions.
              - Per-layer exit accuracies.
        - Finally, returns the dictionary of selected prediction layers (pred_dict) and the tensor
          of modified predictions (new_label) for further evaluations.
    """
    layers = len(pred_local)
    layer_pred, _, _ = gen_pred(pred_local, gt)

    id_pred = [[layer_pred[layer][qid] for layer in range(layers)]
               for qid in gt.keys()]
    print(f'gt_final_labels={id_pred[:10]}')

    # id_pred
    ve = torch.tensor(id_pred)
    # len(ve)
    preds = []
    pred_dict = {}
    for qid in gt.keys():
        pred = [layer_pred[layer][qid] for layer in range(layers)]
        stop_pos = -1 if np.count_nonzero(pred) == 0 else np.argmax(pred)
        preds.append(stop_pos + 1)  # shift one layer
        pred_dict[qid] = stop_pos + 1
    idx = torch.tensor(preds)
    print(f'gt_pos={preds[:10]}')
    print(torch.bincount(idx), torch.bincount(idx) / len(ve))
    cost = torch.tensor([0, 55.581, 109.203, 147.128, 213.685, 266.201])
    # latency, speedup relative to using all 5 layers
    total_cost = cost[idx].mean()
    new_label = torch.cat([torch.zeros(len(ve), 1), ve], 1)
    # the most confident answers for all 5 layers
    total_acc = new_label.gather(1, idx.reshape(-1, 1)).mean() * 100
    print(f'cost={total_cost:.3f} ms, speedup={cost[-1] / total_cost:.2f}, acc={total_acc:.2f}')
    print(torch.count_nonzero(ve, 0))
    # exiting at each layer
    for i in range(layers + 1):
        layer_idx = torch.ones_like(idx).reshape(-1, 1) * i
        layer_acc = new_label.gather(1, layer_idx).mean() * 100
        print(i, layer_acc)
    return pred_dict, new_label


def save_pos_gt(gt_file, gt_pos, out_file='.ee.jsonl'):
    # 'data/vqa/vqa2-local-val.jsonl'
    with open(out_file, 'w') as f:
        for line in open(gt_file):
            gt = json.loads(line)
            qid = gt['question_id']
            pos = gt_pos[qid]
            gt['ee_layer'] =  int(pos)
            f.write(json.dumps(gt) + '\n')


def gen_c_ex(pos, c=2):
    new_pos = {}
    for k, v in pos.items():
        if v > c - 1:
            v = c - 1
        new_pos[k] = v
    return new_pos


# +
pf = '../data/vqa2-local-val-ip0.1.json'
gtf = '../data/vqa/vqa2-local-val.jsonl'

pf = '../data/vqa2-local-test-ip0.1.json'
gtf = '../data/vqa/vqa2-local-test.jsonl'

pf = '../data/vqa2-train-ip0.1.json'
gtf = '../data/vqa/vqa2-train.jsonl'
# -

preds = load_predictions(pf)
gt = load_gt(gtf)
pos, layer_labels = analyze_pred(preds, gt)

# c2 = gen_c_ex(pos, 2)
# save_pos_gt(gtf, c2, '../data/vqa2-local-val-ip0.1-ee-c2')
# save_pos_gt(gtf, c2, '../data/vqa2-local-test-ip0.1-ee-c2.jsonl')
save_pos_gt(gtf, pos, '../data/vqa2-train-ip0.1-ee.jsonl')

# +
# gen balanced train examples
gt_data = [json.loads(line) for line in open('../data/vqa2-train-ip0.1-ee.jsonl')]

layers = 6
gt_list = [[] for _ in range(layers)]
for gti in gt_data:
    gt_list[gti['ee_layer']].append(gti)
    
all_lens = [len(gtl) for gtl in gt_list]
min_len = min(all_lens)
total_len = sum(all_lens)
print(all_lens, min_len)
print([f'{l/total_len*100:.2f}%' for l in all_lens])
# -

sum([i*l for i, l in enumerate(all_lens[1:], 1)])

sum([i*l for i, l in enumerate(all_lens[1:])])

# +
import random

balance_gt = []
for gtl in gt_list[:1]:
    random.shuffle(gtl)
    balance_gt.extend(gtl)
for gtl in gt_list[1:]:
    random.shuffle(gtl)
    bgt = []
    for gti in gtl[:30000]:
        gti['ee_layer'] = 1
        bgt.append(gti)
    balance_gt.extend(bgt)

random.shuffle(balance_gt)
len(balance_gt)
# -

gtb_list = [[] for _ in range(layers)]
for gti in balance_gt:
    gtb_list[gti['ee_layer']].append(gti)
all_lens = [len(gtl) for gtl in gtb_list]
min_len = min(all_lens)
print(all_lens, min_len)
print([f'{l/len(balance_gt)*100:.2f}%' for l in all_lens])

with open('../data/vqa2-train-ip0.1-ee-201k.jsonl', 'w') as f:
    for bgti in balance_gt:
        f.write(json.dumps(bgti) + '\n')


