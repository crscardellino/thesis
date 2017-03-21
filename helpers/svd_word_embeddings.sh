#!/usr/bin/env bash
set -e

cd ..

for corpus in journal regulations sbwcesampled
do
    echo "Getting vectors for $corpus"
    python -m thesis.feature_extraction.word_coocurrence_matrix \
        --size 50 \
        --window_size 5 \
        --min_count 2 \
        /home/ccardellino/sbwce/$corpus \
        ../resources/word_vectors/$corpus.wordvectors.svd.pkl
done
