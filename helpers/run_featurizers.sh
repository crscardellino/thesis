#!/usr/bin/env bash
set -e

cd ..

2>&1 echo "Featurizing sensem"
python -m thesis.feature_extraction \
    ../resources/sensem.conll \
    ../resources/handcrafted_sensem \
    --corpus_name sensem \
    --total_sentences 24157

2>&1 echo "Featurizing semeval"
python -m thesis.feature_extraction \
    ../resources/semeval.conll \
    ../resources/handcrafted_semeval \
    --corpus_name semeval \
    --total_sentences 27132

2>&1 echo "Hashing featurizing sensem"
python -m thesis.feature_extraction \
    ../resources/sensem.conll \
    ../resources/hashed_sensem \
    --corpus_name sensem \
    --total_sentences 24157 \
    --hashing \
    --hashed_features 10000

2>&1 echo "Hashing featurizing semeval"
python -m thesis.feature_extraction \
    ../resources/semeval.conll \
    ../resources/hashed_semeval \
    --corpus_name semeval \
    --total_sentences 27132 \
    --hashing \
    --hashed_features 10000

2>&1 echo "Negative hashing featurizing sensem"
python -m thesis.feature_extraction \
    ../resources/sensem.conll \
    ../resources/negative_hashed_sensem \
    --corpus_name sensem \
    --total_sentences 24157 \
    --hashing \
    --hashed_features 10000 \
    --negative_hash

2>&1 echo "Negative hashing featurizing semeval"
python -m thesis.feature_extraction \
    ../resources/semeval.conll \
    ../resources/negative_hashed_semeval \
    --corpus_name semeval \
    --total_sentences 27132 \
    --hashing \
    --hashed_features 10000 \
    --negative_hash
