#!/usr/bin/env python
# coding: utf8

import os
import logging
import json
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from collections import defaultdict

log = logging.getLogger('CVE_Analyzer')


# ------------------------ PERFORMANCES ------------------------

def _compute_performances(performaces, annotations, entities):
    predictions = [[ent.start_char, ent.end_char, ent.label_] for ent in entities]
    for entry in annotations + predictions:
        if entry in annotations and entry in predictions:
            performaces["tp"] += 1
        elif entry in annotations and entry not in predictions:
            performaces["fn"] += 1
        elif entry not in annotations and entry in predictions:
            performaces["fp"] += 1
        else:
            performaces['tn'] += 1


def _compute_precision(performaces):
    return float(performaces["tp"]) / (performaces["tp"] + performaces["fp"])


def _compute_recall(performaces):
    return float(performaces["tp"]) / (performaces["tp"] + performaces["fn"])


def _compute_f_measure(precision, recall):
    return 2*precision*recall / (precision + recall)


def _compute_accuracy(performaces):
    return float((performaces['tp'] + performaces['tn'])) / \
        float((performaces['tp'] + performaces['tn'] + performaces['fp'] + performaces['fn']))


def _get_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise OSError("Dataset file {} not found".format(dataset_path))
    with open(dataset_path, 'r') as dataset_f:
        dataset = json.load(dataset_f)
    return dataset


def _get_ner_component(nlp):
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')
    return ner


# ------------------------ EXPORTED METHODS ------------------------

def get_train_and_test_sets(dataset_file, split_ratio):
    dataset = _get_dataset(dataset_file)
    random.shuffle(dataset)
    split = int(len(dataset)*split_ratio)
    return dataset[:split], dataset[split:]


def pp_performances(accuracy, precision, recall, f_measure):
    print("\n-------------------------------------------")
    print("PERFORMANCES:")
    print("\nAccuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F-measure: {}".format(f_measure))
    print("\n-------------------------------------------")


def save_model(output_dir, model_name, nlp):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    nlp.meta['name'] = model_name
    nlp.to_disk(output_dir)
    log.debug("Saved model to %s", output_dir)


def get_model(model_path):
    return spacy.load(model_path)


def test(nlp, testset):
    performances = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0
    }
    for description, annotations in testset:
        doc = nlp(description)
        _compute_performances(performances, annotations['entities'], doc.ents)
    performances['accuracy'] = _compute_accuracy(performances)
    performances['precision'] = _compute_precision(performances)
    performances['recall'] = _compute_recall(performances)
    performances['f_measure'] = _compute_f_measure(performances['precision'], performances['recall'])
    return performances


def train(trainset, labels, n_iter, drop_rate):
    nlp = spacy.blank('en')
    ner = _get_ner_component(nlp)
    for label in labels:
        ner.add_label(label)
    optimizer = nlp.begin_training()
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for _ in range(n_iter):
            random.shuffle(trainset)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(trainset, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=drop_rate, losses=losses)
            log.debug('Losses %r', losses)
    return nlp


def get_prediction_for_description(nlp, description):
    doc = nlp(description)
    raw_predictions = [[ent.start_char, ent.end_char, ent.label_] for ent in doc.ents]
    formatted_prediction = defaultdict(list)
    for (start_idx, end_idx, label) in raw_predictions:
        formatted_prediction[label].append(description[start_idx: end_idx])
    return formatted_prediction


def get_default_model():
    return spacy.load(os.path.join(os.path.dirname(__file__), 'model'))


def get_default_dataset():
    return _get_dataset(os.path.join(os.path.dirname(__file__), 'dataset/dataset.json'))
