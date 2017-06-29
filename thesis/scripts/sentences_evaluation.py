#!/usr/bin/env bash

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import pandas as pd
import pickle
import sys

from collections import defaultdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sentences_predictions')
    parser.add_argument('text_sentences')
    parser.add_argument('full_senses_dict')
    parser.add_argument('results_path')

    args = parser.parse_args()

    text_sentences = {}
    with open(args.text_sentences, 'r') as sfin:
        for line in sfin:
            iid, sent = line.strip().split('\t', 1)
            text_sentences[iid] = sent

    with open(args.full_senses_dict, 'rb') as f:
        full_senses_dict = pickle.load(f)

    sentences_prediction = pd.read_csv(args.sentences_predictions)

    results = defaultdict(list)
    final_results = []

    for (algorithm, lemma), sdf in sentences_prediction.groupby(('algorithm', 'lemma'), sort=False):
        classes = dict(sdf.groupby('class_label').first()['predicted_target_mapping'])
        senses = sorted(full_senses_dict[lemma].items())
        correct = 0

        for idx, sentence in sdf.sample(frac=1.).reset_index(drop=True).iterrows():
            print('*' * 50 + '\n%s' % text_sentences[sentence.instance], file=sys.stderr)
            print('*' * 50 + '\nSelect the sense for the previous sentence:', file=sys.stderr)

            for idx, (sense, description) in enumerate(senses):
                print('%d ) %s: %s' % (idx, sense, description), file=sys.stderr)
            while True:
                sense = input('Sense: ')
                try:
                    sense = int(sense)
                except ValueError:
                    continue
                if -2 <= sense < len(senses):
                    break

            print(file=sys.stderr)

            if sense >= 0 and senses[sense][0] not in classes:
                classes[senses[sense][0]] = len(classes)

            if sense >= 0:
                results[(algorithm, lemma)].append({
                    'algorithm': algorithm,
                    'lemma': lemma,
                    'instance': sentence.instance,
                    'true': classes[senses[sense][0]],
                    'prediction': sentence.predicted_target_mapping
                })

                if classes[senses[sense][0]] == sentence.predicted_target_mapping:
                    correct += 1
            elif sense == -1:
                results[(algorithm, lemma)].append({
                    'algorithm': algorithm,
                    'lemma': lemma,
                    'instance': sentence.instance,
                    'true': sense,
                    'prediction': sentence.predicted_target_mapping
                })

            if sense == -2:
                break

            print('Number of evaluated samples so far: %d' % len(results[(algorithm, lemma)]), file=sys.stderr)
            print('Accuracy so far: %.2f' % (correct / len(results[(algorithm, lemma)])), file=sys.stderr)

        if len(results[(algorithm, lemma)]) > 0:
            final_results.append(pd.DataFrame(results[(algorithm, lemma)],
                                              columns=['algorithm', 'lemma', 'instance', 'true', 'prediction']))
            final_results[-1].insert(2, 'final_accuracy', (correct / len(results[(algorithm, lemma)])))

    final_results = pd.concat(final_results, ignore_index=True)
    final_results['true'] = final_results['true'].astype(int)
    final_results['prediction'] = final_results['prediction'].astype(int)
    final_results.to_csv(args.results_path, float_format='%.2f')
