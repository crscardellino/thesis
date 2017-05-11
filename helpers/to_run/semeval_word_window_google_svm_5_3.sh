#!/usr/bin/env bash
set -e
cd ..

cd ..

python -m thesis.classification \
    ../resources/word_window/semeval/train_dataset.npz \
    ../resources/word_window/semeval/test_dataset.npz \
    ../results/semisupervised/semeval_svm_word_window_NA_google.csv \
    --classifier svm \
    --word_vectors_model_path ../resources/word_vectors/google.bin.gz \
    --vector_domain google \
    --min_count 2 \
    --corpus_name semeval \
    --representation word_window

python -m thesis.classification \
    ../resources/word_window/semeval/train_dataset.npz \
    ../resources/word_window/semeval/test_dataset.npz \
    ../results/semisupervised/semeval_svm_word_window_NA_google_5_3.csv \
    --classifier svm \
    --splits 5 \
    --folds 3 \
    --word_vectors_model_path ../resources/word_vectors/google.bin.gz \
    --vector_domain google \
    --min_count 2 \
    --corpus_name semeval \
    --representation word_window
