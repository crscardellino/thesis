#!/usr/bin/env ruby

Dir.chdir ".."

CLASSIFIERS = %w(baseline naive_bayes decision_tree log svm mlp)
REPRESENTATIONS = %w(hashed handcrafted feature_selection)
CORPORA = %w(semeval sensem)
LAYERS = %w(100 250 500 250_100 500_250_100)
MAX_FEATURES = 10000

BASE_CMD = "python -m thesis.classification "
BASE_CMD.concat("../resources/%{representation}/%{corpus}/train_dataset.npz ")
BASE_CMD.concat("../resources/%{representation}/%{corpus}/test_dataset.npz ")
BASE_CMD.concat("../results/supervised/%{results_file}.csv ")
BASE_CMD.concat("--classifier %{classifier} ")
BASE_CMD.concat("--corpus_name %{corpus} ")
BASE_CMD.concat("--representation %{representation} ")

BASE_RESULTS = "%{corpus}_%{classifier}_%{representation}_%{layers}_%{domain}_%{splits_and_folds}"

SPLITS = 5
FOLDS = 3

CLASSIFIERS.product(REPRESENTATIONS, CORPORA).each do |classifier, representation, corpus|
    next if representation == "handcrafted" and classifier == "mlp"

    cmd = BASE_CMD
    cmd_hash = {classifier: classifier, representation: representation, corpus: corpus}
    results = BASE_RESULTS
    results_hash = Hash.new("NA")
    results_hash[:classifier] = classifier
    results_hash[:corpus] = corpus
    results_hash[:representation] = representation

    if representation == "feature_selection" then
        cmd_hash[:representation] = "handcrafted"
        cmd += "--max_features #{MAX_FEATURES} "
    end

    if classifier == "mlp" then
        cmd += "--layers #{LAYERS[0]} "
        results_hash[:layers] = LAYERS[0]
    end

    cmd_hash[:results_file] = results % results_hash

    STDERR.puts cmd % cmd_hash
    raise "Error on command" unless system cmd % cmd_hash

    cmd += "--splits #{SPLITS} --folds #{FOLDS} "
    results_hash[:splits_and_folds] = "#{SPLITS}_#{FOLDS}"
    cmd_hash[:results_file] = results % results_hash

    if classifier != "baseline" and classifier != "naive_bayes" then
        STDERR.puts cmd % cmd_hash
        raise "Error on command" unless system cmd % cmd_hash
    end
end

REPRESENTATIONS.product(CORPORA, LAYERS.drop(1)).each do |representation, corpus, layers|
    next if representation == "handcrafted"
    classifier = "mlp"

    cmd = BASE_CMD
    cmd_hash = {classifier: classifier, representation: representation, corpus: corpus}
    results = BASE_RESULTS
    results_hash = Hash.new("NA")
    results_hash[:classifier] = classifier
    results_hash[:corpus] = corpus
    results_hash[:representation] = representation

    if representation == "feature_selection" then
        cmd_hash[:representation] = "handcrafted"
        cmd += "--max_features #{MAX_FEATURES} "
    end

    cmd += "--layers #{layers.gsub("_", " ")} "
    results_hash[:layers] = layers

    cmd_hash[:results_file] = results % results_hash

    STDERR.puts cmd % cmd_hash
    raise "Error on command" unless system cmd % cmd_hash

    cmd += "--splits #{SPLITS} --folds #{FOLDS} "
    results_hash[:splits_and_folds] = "#{SPLITS}_#{FOLDS}"
    cmd_hash[:results_file] = results % results_hash

    if classifier != "baseline" and classifier != "naive_bayes" then
        STDERR.puts cmd % cmd_hash
        raise "Error on command" unless system cmd % cmd_hash
    end
end
