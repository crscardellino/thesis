#!/usr/bin/env zsh
set -e

cd ..

python -m thesis.classification.semisupervised \\
    --word_vector_model_path ../resources/word_vectors/journal.wordvectors.bin.gz \\
    --classifier mlp --layers 
