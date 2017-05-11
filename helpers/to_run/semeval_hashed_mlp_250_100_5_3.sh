#!/usr/bin/env bash
set -e
cd ..

cd ..

python -m thesis.classification \
    ../resources/hashed/semeval/train_dataset.npz \
    ../resources/hashed/semeval/test_dataset.npz \
    ../results/semisupervised/semeval_mlp_hashed_250_100_NA.csv \
    --classifier mlp \
    --layers 250 100 \
    --min_count 2 \
    --corpus_name semeval \
    --representation hashed

python -m thesis.classification \
    ../resources/hashed/semeval/train_dataset.npz \
    ../resources/hashed/semeval/test_dataset.npz \
    ../results/semisupervised/semeval_mlp_hashed_250_100_NA_5_3.csv \
    --classifier mlp \
    --layers 250 100 \
    --splits 5 \
    --folds 3 \
    --min_count 2 \
    --corpus_name semeval \
    --representation hashed
