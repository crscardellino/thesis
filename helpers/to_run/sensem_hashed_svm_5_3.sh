#!/usr/bin/env bash
set -e
cd ..

cd ..

python -m thesis.classification \
    ../resources/hashed/sensem/train_dataset.npz \
    ../resources/hashed/sensem/test_dataset.npz \
    ../results/semisupervised/sensem_svm_hashed_NA_NA.csv \
    --classifier svm \
    --min_count 2 \
    --corpus_name sensem \
    --representation hashed

python -m thesis.classification \
    ../resources/hashed/sensem/train_dataset.npz \
    ../resources/hashed/sensem/test_dataset.npz \
    ../results/semisupervised/sensem_svm_hashed_NA_NA_5_3.csv \
    --classifier svm \
    --splits 5 \
    --folds 3 \
    --min_count 2 \
    --corpus_name sensem \
    --representation hashed
