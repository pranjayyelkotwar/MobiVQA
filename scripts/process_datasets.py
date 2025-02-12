#!/usr/bin/env python3
# coding: utf-8
import json
import os
import re

"""
Modified version with added documentation and print statements for debugging purposes.
Based on original code from:
https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/392520640e3d9aed0009ddfe207901757b10b9a6/dataset.py
"""

# Contractions mapping to expand contractions
contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've"
}

# Mapping numbers to digits for manual conversion.
manual_map = {
    'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10'
}

articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>',
         '<', '@', '`', ',', '?', '!']

def get_score(occurences):
    """
    Calculates the score based on answer frequency.
    
    Args:
        occurences (int): Number of times the answer appears.
        
    Returns:
        float: Score assigned to the answer.
    """
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1

def process_punctuation(inText):
    """
    Removes or replaces punctuation in a given text.
    
    Args:
        inText (str): Input text string.
        
    Returns:
        str: Text processed without unwanted punctuation.
    """
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(comma_strip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText

def process_digit_article(inText):
    """
    Processes digits and articles in the input text by converting number words
    to digits and removing articles.
    
    Args:
        inText (str): Input text string.
        
    Returns:
        str: Processed text.
    """
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def multiple_replace(text, wordDict):
    """
    Performs multiple search and replace operations over a string.
    
    Args:
        text (str): The text to operate on.
        wordDict (dict): A dictionary with keys to search for and values to replace with.
        
    Returns:
        str: The text after replacements.
    """
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text

def preprocess_answer(answer):
    """
    Preprocess the answer string by processing punctuation, digits,
    and articles.
    
    Args:
        answer (str): The original answer.
        
    Returns:
        str: Preprocessed version of the answer.
    """
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer

# Mapping of answer/question/image id fields across different datasets.
ans_field = {
    'vqa2': 'multiple_choice_answer',
    'vg': 'answer',
    'vizwiz': 'answers',
}
q_id_field = {
    'vqa2': 'question_id',
    'vg': 'qa_id',
    'vizwiz': 'image',
}
img_id_field = {
    'vqa2': 'image_id',
    'vg': 'image_id',
    'vizwiz': 'image',
}

def filter_vw_answers(qa_data, min_occurence, dataset='vw'):
    """
    Filters vizwiz answers based on a minimum occurrence,
    preprocesses them, and prints the count of valid answers.
    
    Args:
        qa_data (list): List of QA data entries.
        min_occurence (int): Minimum occurrence threshold.
        dataset (str): Name of the dataset.
    
    Returns:
        dict: Mapping of answers meeting the threshold with their corresponding image sets.
    """
    print("Filtering vizwiz answers with minimum occurrence:", min_occurence)
    occurence = {}

    for ans_entry in qa_data:
        answers = ans_entry['answers']
        for ans in answers:
            gtruth = ans['answer']
            gtruth = preprocess_answer(gtruth)
            if gtruth not in occurence:
                occurence[gtruth] = set()
            occurence[gtruth].add(ans_entry['image'])
    # Remove answers that do not meet the minimum occurrence.
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)
    print('Num of vizwiz answers that appear >= %d times: %d' % (min_occurence, len(occurence)))
    return occurence

def filter_answers(qa_data, min_occurence, dataset='vqa2'):
    """
    Filters answers for VQA2 or VG dataset based on frequency threshold.
    
    Args:
        qa_data (list): List of QA data entries.
        min_occurence (int): Minimum frequency required.
        dataset (str): The dataset type.
        
    Returns:
        dict: Filtered answer occurrence dictionary.
    """
    print("Filtering answers for dataset", dataset, "with minimum occurrence:", min_occurence)
    occurence = {}

    for ans_entry in qa_data:
        gtruth = ans_entry[ans_field[dataset]]
        gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry[q_id_field[dataset]])
    # Remove answers that do not meet the threshold.
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)
    print('Num of answers that appear >= %d times: %d' % (min_occurence, len(occurence)))
    return occurence

def create_ans2label(occurrence, name, cache_dir='data/cache'):
    """
    Creates a mapping from answer to unique label and saves the mappings to cache.
    
    Args:
        occurrence (dict): Dictionary of answer occurrences.
        name (str): Dataset name used for naming cache files.
        cache_dir (str): Directory to store cached files.
        
    Returns:
        dict: Mapping from answer to label.
    """
    print("Creating answer-to-label mapping for", name)
    ans2label = {}
    label2ans = []
    label = 0
    for answer in sorted(occurrence):
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, name + '_ans2label.json')
    json.dump(ans2label, open(cache_file, 'w'), indent=2)
    print("Saved ans2label mapping to", cache_file)
    
    cache_file = os.path.join(cache_dir, name + '_label2ans.json')
    json.dump(label2ans, open(cache_file, 'w'), indent=2)
    print("Saved label2ans mapping to", cache_file)
    return ans2label

def compute_target(qa_data, id_questions, ans2label, dataset='vqa2', split='val2014'):
    """
    Processes QA data and writes the computed target to a jsonl file.
    
    Args:
        qa_data (list): List of QA data entries.
        id_questions (dict): Mapping of question id to question text.
        ans2label (dict): Answer to label mapping.
        dataset (str): Dataset type.
        split (str): Dataset split name.
    """
    cache_filename = f'data/cache/{dataset}-{split}.jsonl'
    print("Computing target and saving to", cache_filename)
    with open(cache_filename, 'w') as f:
        for entry in qa_data:
            labels = None

            if dataset == 'vg':
                ans = entry.get('answer', None)
                if ans:
                    ans_ = preprocess_answer(ans)
                    labels = {ans_: 1 if ans_ in ans2label else 0}
            elif dataset == 'vqa2' or dataset == 'vizwiz':
                answers = entry.get('answers', None)
                if answers:
                    answer_count = {}
                    for answer in answers:
                        answer_ = answer['answer']
                        answer_count[answer_] = answer_count.get(answer_, 0) + 1

                    labels = {}
                    for answer in answer_count:
                        if answer not in ans2label:
                            continue
                        score = get_score(answer_count[answer])
                        labels[answer] = score
            else:
                raise ValueError('not supported dataset type: {}'.format(dataset))

            qid = entry[q_id_field[dataset]]
            img_id = entry[img_id_field[dataset]]
            if dataset == 'vqa2':
                q_sent = id_questions[qid]
                image_id = 'COCO_{}_{:012d}'.format(split, img_id)
            else:  # vg, vizwiz or no id
                q_sent = entry['question']
                image_id = img_id

            target = {
                'question_id': qid,
                'sent': q_sent,
                'img_id': image_id,
            }
            if labels is not None:  # caveat! label can be empty
                target['label'] = labels
            ans_type = entry.get('answer_type', '')
            if ans_type:
                target['answer_type'] = ans_type
            f.write(json.dumps(target, ensure_ascii=False))
            f.write('\n')
    print("Finished computing target for", dataset, split)

def process_vqa2():
    """
    Processes the VQA2 dataset: loads training, validation and test files,
    computes answer mappings and target files.
    """
    print("Starting processing for VQA2 dataset...")
    train_answer_file = 'data/original/v2_mscoco_train2014_annotations.json'
    train_answers = json.load(open(train_answer_file))['annotations']
    train_question_file = 'data/original/v2_OpenEnded_mscoco_train2014_questions.json'
    train_questions = json.load(open(train_question_file))['questions']
    train_id_questions = {i['question_id']: i['question'] for i in train_questions}
    print('Loaded COCO train2014 data.')

    val_answer_file = 'data/original/v2_mscoco_val2014_annotations.json'
    val_answers = json.load(open(val_answer_file))['annotations']
    val_question_file = 'data/original/v2_OpenEnded_mscoco_val2014_questions.json'
    val_questions = json.load(open(val_question_file))['questions']
    val_id_questions = {i['question_id']: i['question'] for i in val_questions}
    print('Loaded COCO val2014 data.')

    train_val_answers = train_answers + val_answers
    train_val_occurrence = filter_answers(train_val_answers, 9, 'vqa2')
    trainval_ans2label = create_ans2label(train_val_occurrence, 'vqa2')

    test_question_file = 'data/original/v2_OpenEnded_mscoco_test2015_questions.json'
    test_questions = json.load(open(test_question_file))['questions']
    print('Loaded COCO test2015 data.')

    cache_root = 'data/cache'
    os.makedirs(cache_root, exist_ok=True)
    compute_target(train_answers, train_id_questions, trainval_ans2label, dataset='vqa2', split='train2014')
    print('Saved COCO train2014 target.')
    
    compute_target(val_answers, val_id_questions, trainval_ans2label, dataset='vqa2', split='val2014')
    print('Saved COCO val2014 target.')
    
    test_id_questions = {i['question_id']: i['question'] for i in test_questions}
    compute_target(test_questions, test_id_questions, {}, dataset='vqa2', split='test2015')
    print('Saved COCO test2015 target.')

def process_vg():
    """
    Processes the Visual Genome (VG) dataset: loads VG question-answer file,
    computes answer mappings and target file.
    """
    print("Starting processing for VG dataset...")
    vg_qa_file = 'data/original/question_answers.json'
    vg_qa = json.load(open(vg_qa_file))
    print('Loaded Visual Genome data.')
    vg_answers = [qai for qas in vg_qa for qai in qas['qas']]
    vg_occurrence = filter_answers(vg_answers, 9, dataset='vg')
    vg_ans2label = create_ans2label(vg_occurrence, 'vg')
    compute_target(vg_answers, {}, vg_ans2label, dataset='vg', split='all')
    print('Saved VG targets, processing complete.')

def process_vizwiz():
    """
    Processes the VizWiz dataset: loads train, validation, and test files,
    computes answer mappings and target files.
    """
    print("Starting processing for VizWiz dataset...")
    vw_file = 'data/original/Annotations/train.json'
    vw_train_qa = json.load(open(vw_file))
    print('Loaded VizWiz train data.')

    vw_val_file = 'data/original/Annotations/val.json'
    vw_val_qa = json.load(open(vw_val_file))
    print('Loaded VizWiz validation data.')

    vw_qa = vw_train_qa + vw_val_qa
    vw_occurrence = filter_vw_answers(vw_qa, 5, dataset='vizwiz')
    vw_ans2label = create_ans2label(vw_occurrence, 'vizwiz')
    
    compute_target(vw_train_qa, {}, vw_ans2label, dataset='vizwiz', split='train')
    print('Saved VizWiz train targets.')
    
    compute_target(vw_val_qa, {}, vw_ans2label, dataset='vizwiz', split='val')
    print('Saved VizWiz validation targets.')

    vw_test_file = 'data/original/Annotations/test.json'
    vw_test_qa = json.load(open(vw_test_file))
    print('Loaded VizWiz test data.')
    compute_target(vw_test_qa, {}, vw_ans2label, dataset='vizwiz', split='test')
    print('Saved VizWiz test targets. VizWiz processing complete.')

if __name__ == '__main__':
    # Uncomment the corresponding process functions you want to run:
    # process_vqa2()
    # process_vg()
    process_vizwiz()
