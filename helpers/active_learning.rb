#!/usr/bin/env ruby

Dir.chdir '..'

CLASSIFIERS = %w(svm mlp)
REPRESENTATIONS = %w(hashed word_window)
CORPORA = %w(sensem)
LAYERS = %w(100 250_100 500_250_100)
CANDIDATES_LIMIT = 10
CANDIDATES_SELECTION = %w(min max random)
VECTOR_DOMAIN = %w(journal sbwce)
FOLDS = 3

BASE_CMD = 'python -W ignore -m thesis.classification.active_learning '
BASE_CMD.concat('../resources/%{representation}/%{corpus} ')
BASE_CMD.concat('../results/active_learning/%{results_base} ')
BASE_CMD.concat('--simulation_indices_path ../resources/active_learning/%{corpus}.npz')
BASE_CMD.concat('--classifier %{classifier} ')
BASE_CMD.concat('--candidates_selection %{selection} ')
BASE_CMD.concat("--candidates_limit #{CANDIDATES_LIMIT} ")
BASE_CMD.concat('--corpus_name %{corpus} ')
BASE_CMD.concat('--representation %{representation} ')
BASE_CMD.concat('--vector_domain %{vector_domain} ')

BASE_RESULTS = '%{corpus}_%{classifier}_%{representation}_%{selection}_%{layers}_%{domain}_%{folds}'

CLASSIFIERS.product(
    REPRESENTATIONS, CORPORA, CANDIDATES_SELECTION).each do |classifier, representation, corpus, selection|
  cmd = BASE_CMD
  cmd_hash = {classifier: classifier, representation: representation, corpus: corpus, selection: selection}
  results = BASE_CMD
  results_hash = Hash.new('NA')
  results_hash[:classifier] = classifier
  results_hash[:corpus] = corpus
  results_hash[:representation] = representation
  results_hash[:selection] = selection

  if classifier == 'mlp'
    cmd += "--layers #{LAYERS[0]} "
    results_hash[:layers] = LAYERS[0]
  end

  cmd_hash[:results_base] = results % results_hash

  STDERR.puts cmd % cmd_hash
  raise 'Error on command' unless system cmd % cmd_hash

  cmd += "--folds #{FOLDS} "
  results_hash[:folds] = "#{FOLDS}"
  cmd_hash[:results_base] = results % results_hash

  STDERR.puts cmd % cmd_hash
  raise 'Error on command' unless system cmd % cmd_hash
end

#REPRESENTATIONS.product(CORPORA, LAYERS.drop(1)).each do |representation, corpus, layers|
#  next if representation == 'handcrafted'
#  classifier = 'mlp'
#
#  cmd = BASE_CMD
#  cmd_hash = {classifier: classifier, representation: representation, corpus: corpus}
#  results = BASE_RESULTS
#  results_hash = Hash.new('NA')
#  results_hash[:classifier] = classifier
#  results_hash[:corpus] = corpus
#  results_hash[:representation] = representation
#
#  if representation == 'feature_selection'
#      cmd_hash[:representation] = 'handcrafted'
#      cmd += '--max_features #{MAX_FEATURES} '
#  end
#
#  cmd += '--layers #{layers.gsub('_', ' ')} '
#  results_hash[:layers] = layers
#
#  cmd_hash[:results_file] = results % results_hash
#
#  STDERR.puts cmd % cmd_hash
#  raise 'Error on command' unless system cmd % cmd_hash
#
#  cmd += '--splits #{SPLITS} --folds #{FOLDS} '
#  results_hash[:splits_and_folds] = '#{SPLITS}_#{FOLDS}'
#   cmd_hash[:results_file] = results % results_hash
#
#   if classifier != 'baseline' and classifier != 'naive_bayes'
#       STDERR.puts cmd % cmd_hash
#       raise 'Error on command' unless system cmd % cmd_hash
#   end
# end
