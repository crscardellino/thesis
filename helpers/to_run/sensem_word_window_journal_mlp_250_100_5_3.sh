#!/usr/bin/env bash
set -e
cd ..

cd ..

python -m thesis.classification \
    ../resources/word_window/sensem/train_dataset.npz \
    ../resources/word_window/sensem/test_dataset.npz \
    ../results/semisupervised/sensem_mlp_word_window_250_100_journal.csv \
    --classifier mlp \
    --layers 250 100 \
    --word_vectors_model_path ../resources/word_vectors/journal.bin.gz \
    --vector_domain journal \
    --min_count 2 \
    --corpus_name sensem \
    --representation word_window

python -m thesis.classification \
    ../resources/word_window/sensem/train_dataset.npz \
    ../resources/word_window/sensem/test_dataset.npz \
    ../results/semisupervised/sensem_mlp_word_window_250_100_journal_5_3.csv \
    --classifier mlp \
    --layers 250 100 \
    --splits 5 \
    --folds 3 \
    --word_vectors_model_path ../resources/word_vectors/journal.bin.gz \
    --vector_domain journal \
    --min_count 2 \
    --corpus_name sensem \
    --representation word_window
