import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['WANDB_SILENT'] = 'true'

import time
import json
import click
import logging

import numpy as np
import pandas as pd

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

from utils import get_n_chunks


logging.getLogger('transformers').setLevel(logging.ERROR)



def clean_labels(labels):
    print('Cleaning Labels...\n')

    labels_fixed = []

    for i, label_set in enumerate(labels):
        # convert labels from string to list
        label_set = label_set.lower().replace('[', '').replace(']', '').replace('"', '').split(', ')

        # keep only unique labels
        label_set = set(label_set)

        cur_labels = []
        for label in label_set:
            cur_labels.append(label)

        labels_fixed.append(cur_labels)

    return pd.Series(labels_fixed)


def get_target_labels():
    with open(os.path.join('..', 'labels.json')) as f:
        label_mapping = json.load(f)

    aliases = set([k for k in label_mapping.keys()])
    target_labels = [v for v in label_mapping.values()]

    return aliases, target_labels, label_mapping


def disambiguate_labels(labels_dict, disambiguate='keep_most_probable'):
    print('Disambiguating Labels...\n')

    assert disambiguate in ['keep_most_probable', 'drop_all']

    cleaned_dict = {}

    for key in labels_dict:
        if len(labels_dict[key]) > 1:
            if disambiguate == 'drop_all':
                cleaned_dict[key] = None
            else:
                max_likelihood = 0
                best_match = None

                for label, likelihood in labels_dict[key]:
                    if likelihood > max_likelihood:
                        max_likelihood = likelihood
                        best_match = label
                
                cleaned_dict[key] = best_match
        else:
            label, likelihood = labels_dict[key][0]
            cleaned_dict[key] = label
    
    return cleaned_dict


def map_labels(label_series, mapping):
    print('Mappping Labels...\n')

    mapped_labels = []
    
    for i, label_list in enumerate(label_series):
        temp_labels = []

        for l in label_list:
            try:
                temp_labels.append(mapping[l])
            except:
                pass
        
        if temp_labels:
            mapped_labels.append(temp_labels)
        else:
            mapped_labels.append(['undefined'])

    return pd.Series(mapped_labels)


def get_mapping(paraphrase_candidates, paraphrase_likelihood, threshold=.5):
    label_mapping = {}
    
    for pair, pair_likelihood in zip(paraphrase_candidates, paraphrase_likelihood):
        if pair_likelihood > threshold:
            target_l, real_l = pair[0], pair[1]
            
            try:
                label_mapping[real_l].append((LABELS[target_l], pair_likelihood))
            except:
                label_mapping[real_l] = []
                label_mapping[real_l].append((LABELS[target_l], pair_likelihood))

    return label_mapping


def make_combinations(targets, observed):
    print('Creating Label Combinations...\n')

    combinations = []
    
    for t in targets:
        for o in observed:
            combinations.append([t, o])

    return combinations


def check_paraphrase(inputs, low_memory=True, chunk_size=1000):
    print('Checking for Paraphrase...\n') 

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased-finetuned-mrpc', output_loading_info=False)
    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased-finetuned-mrpc', output_loading_info=False)    

    if low_memory:
        n_chunks = get_n_chunks(inputs, chunk_size)
        
        paraphrase_likelihood = np.array([])

        for i in range(n_chunks):
            from time import time
            start = time()

            print(f'Chunk: {i+1}/{n_chunks}\r', end='')

            try:
                cur_inputs = inputs[i*chunk_size:(i+1)*chunk_size]
            except IndexError:
                cur_inputs = inputs[i*chunk_size:]

            cur_inputs = tokenizer(cur_inputs, 
                                   padding=True, 
                                   truncation=True, 
                                   return_tensors='tf')

            logits = model(cur_inputs)[0]
            outputs = tf.nn.softmax(logits, axis=1).numpy()

            paraphrase_likelihood = np.append(paraphrase_likelihood, outputs[:, 1])

            print(f'Time elapsed for chunk {i}: {time() - start}sec')
    else:
        cur_inputs = tokenizer(cur_inputs, 
                               padding=True, 
                               truncation=True, 
                               return_tensors='tf')

        logits = model(cur_inputs)[0]
        outputs = tf.nn.softmax(logits, axis=1).numpy()

        paraphrase_likelihood = outputs[:, 1]

    return paraphrase_likelihood


def main(label_series):
    global LABELS
    aliases, target_labels, LABELS = get_target_labels()

    label_series = clean_labels(label_series)
    unique_labels = label_series.explode().value_counts().keys().values

    paraphrase_candidates = make_combinations(aliases, unique_labels)
    paraphrase_likelihood = check_paraphrase(paraphrase_candidates)

    label_mapping = get_mapping(paraphrase_candidates, paraphrase_likelihood)
    label_mapping = disambiguate_labels(label_mapping)
    label_series = map_labels(label_series, label_mapping)


    if 'undefined' not in target_labels:
        target_labels.append('undefined')

    return label_series, target_labels


@click.command()
@click.option('--limit', '-L', default=None, type=int)
def cli(limit):
    label_series = pd.read_pickle('data/github_raw.pkl')['labels'][:limit]
    label_series, LABELS = main(label_series)



if __name__ == '__main__':
    cli()
