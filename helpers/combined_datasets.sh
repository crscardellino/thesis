#!/usr/bin/env zsh
set -e

cd ..

dataset=combined_dataset
directory=../resources/hashed_sensem
directory_extra=../resources/word_window_sensem
word_vectors_model=../resources/word_vectors/journal.wordvectors.bin.gz
results=../results/experiment_combined

>&2 echo "Running classifier svm"
python -m thesis.classification \
    $directory/train_dataset.npz \
    $directory/test_dataset.npz \
    $results/svm_combined.csv \
    --classifier svm \
    --word_vectors_model_path $word_vectors_model \
    --train_dataset_extra $directory_extra/train_dataset.npz \
    --test_dataset_extra $directory_extra/test_dataset.npz

>&2 echo "Running classifier mlp for $dataset"
python -m thesis.classification \
    $directory/train_dataset.npz \
    $directory/test_dataset.npz \
    $results/mlp_combined.csv \
    --classifier mlp \
    --word_vectors_model_path $word_vectors_model \
    --train_dataset_extra $directory_extra/train_dataset.npz \
    --test_dataset_extra $directory_extra/test_dataset.npz \
    --layers 200

for splits in 3
do
    for folds in 3 10
    do
        if [[ $(( splits * folds )) -gt 30 ]]
        then
            continue
        fi

        >&2 echo "Running classifier svm for $splits splits and $folds folds"
        python -m thesis.classification \
            $directory/train_dataset.npz \
            $directory/test_dataset.npz \
            $results/svm_${splits}_splits_${folds}_folds.csv \
            --classifier svm \
            --word_vectors_model_path $word_vectors_model \
            --train_dataset_extra $directory_extra/train_dataset.npz \
            --test_dataset_extra $directory_extra/test_dataset.npz \
            --splits $splits \
            --folds $folds \
            --ensure_minimum

        >&2 echo "Running classifier mlp for $dataset for $splits splits and $folds folds"
        python -m thesis.classification \
            $directory/train_dataset.npz \
            $directory/test_dataset.npz \
            $results/mlp_${splits}_splits_${folds}_folds.csv \
            --classifier mlp \
            --word_vectors_model_path $word_vectors_model \
            --train_dataset_extra $directory_extra/train_dataset.npz \
            --test_dataset_extra $directory_extra/test_dataset.npz \
            --layers 200 \
            --splits $splits \
            --folds $folds \
            --ensure_minimum
    done
done
