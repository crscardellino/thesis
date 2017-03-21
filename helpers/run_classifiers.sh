#!/usr/bin/env zsh
set -e

vector_ext=${1:-svd.pkl}

cd ..

find ../resources -type d -name "word_window_sensem" | sort | while read directory
do
    dataset=$(basename $directory)

    for wv in journal regulations sbwcesampled
    do
        >&2 echo "Running classifier svm for $dataset"
        python -m thesis.classification \
            $directory/train_dataset.npz \
            $directory/test_dataset.npz \
            ../results/experiment_word_vectors/svm_${dataset}_${wv}_${vector_ext}.csv \
            --classifier svm \
            --word_vectors_model_path ../resources/word_vectors/${wv}.wordvectors.${vector_ext}

        >&2 echo "Running classifier mlp for $dataset"
        python -m thesis.classification \
            $directory/train_dataset.npz \
            $directory/test_dataset.npz \
            ../results/experiment_word_vectors/mlp_1800_${dataset}_${wv}_${vector_ext}.csv \
            --classifier mlp \
            --word_vectors_model_path ../resources/word_vectors/${wv}.wordvectors.${vector_ext} \
            --layers 1800

        for splits in 3
        do
            for folds in 3 10
            do
                if [[ $(( splits * folds )) -gt 30 ]]
                then
                    continue
                fi

                >&2 echo "Running classifier svm for $dataset for $splits splits and $folds folds"
                python -m thesis.classification \
                    $directory/train_dataset.npz \
                    $directory/test_dataset.npz \
                    ../results/experiment_word_vectors/svm_${dataset}_${splits}_splits_${folds}_folds_${wv}_${vector_ext}.csv \
                    --classifier svm \
                    --word_vectors_model_path ../resources/word_vectors/${wv}.wordvectors.${vector_ext} \
                    --splits $splits \
                    --folds $folds \
                    --ensure_minimum

                >&2 echo "Running classifier mlp for $dataset for $splits splits and $folds folds"
                python -m thesis.classification \
                    $directory/train_dataset.npz \
                    $directory/test_dataset.npz \
                    ../results/experiment_word_vectors/mlp_1800_${dataset}_${splits}_splits_${folds}_folds_${wv}_${vector_ext}.csv \
                    --classifier mlp \
                    --word_vectors_model_path ../resources/word_vectors/${wv}.wordvectors.${vector_ext} \
                    --layers 1800 \
                    --splits $splits \
                    --folds $folds \
                    --ensure_minimum
            done
        done
    done
done
