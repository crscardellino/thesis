#!/usr/bin/env zsh
set -e

cd ..

find ../resources -type d -not -name "resources" | sort | while read directory
do
    dataset=$(basename $directory)

    for classifier in baseline decision_tree log svm
    do
        >&2 echo "Running classifier $classifier for $dataset"
        python -m thesis.classification \
            $directory/train_dataset.npz \
            $directory/test_dataset.npz \
            ../results/${classifier}_${dataset}.csv \
            --classifier $classifier \
            --max_features 10000
    done

    # We run the non negative datasets for naive bayes
    if ! [[ $dataset =~ "negative" ]]
    then
        >&2 echo "Running classifier naive_bayes for $dataset"
        python -m thesis.classification \
            $directory/train_dataset.npz \
            $directory/test_dataset.npz \
            ../results/naive_bayes_${dataset}.csv \
            --classifier naive_bayes \
            --max_features 10000
    fi

    >&2 echo "Running classifier mlp for $dataset"
    python -m thesis.classification \
        $directory/train_dataset.npz \
        $directory/test_dataset.npz \
        ../results/mlp_5000_${dataset}.csv \
        --classifier mlp \
        --max_features 10000 \
        --layers 5000
done
