#!/usr/bin/env bash
set -e
cd ..

cd ..

python -m thesis.classification \
    ../resources/hashed/sensem/train_dataset.npz \
    ../resources/hashed/sensem/test_dataset.npz \
    ../results/semisupervised/sensem_mlp_hashed_100_NA.csv \
    --classifier mlp \
    --layers 100 \
    --min_count 2 \
    --corpus_name sensem \
    --representation hashed

python -m thesis.classification \
    ../resources/hashed/sensem/train_dataset.npz \
    ../resources/hashed/sensem/test_dataset.npz \
    ../results/semisupervised/sensem_mlp_hashed_100_NA_5_3.csv \
    --classifier mlp \
    --layers 100 \
    --splits 5 \
    --folds 3 \
    --min_count 2 \
    --corpus_name sensem \
    --representation hashed
