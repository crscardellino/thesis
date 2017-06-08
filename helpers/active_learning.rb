#!/usr/bin/env ruby

Dir.chdir '..'

CLASSIFIERS = %w(svm mlp)
REPRESENTATIONS = %w(hashed word_window)
CORPORA = %w(sensem)
LAYERS = [nil] + %w(100 250_100)
CANDIDATES_LIMIT = 5
CANDIDATES_SELECTION = %w(min max random)
VECTOR_DOMAIN = [nil] + %w(journal sbwce)
FOLDS = 3

BASE_CMD = 'python -W ignore -m thesis.classification.active_learning '
BASE_CMD.concat('../resources/%{representation}/%{corpus} ')
BASE_CMD.concat('../results/active_learning/%{results_base} ')
BASE_CMD.concat('--simulation_indices_path ../resources/active_learning/%{corpus}.npz ')
BASE_CMD.concat('--classifier %{classifier} ')
BASE_CMD.concat('--candidates_selection %{selection} ')
BASE_CMD.concat("--candidates_limit #{CANDIDATES_LIMIT} ")
BASE_CMD.concat('--corpus_name %{corpus} ')
BASE_CMD.concat('--representation %{representation} ')

BASE_RESULTS = '%{corpus}_%{classifier}_%{representation}_%{selection}_%{layers}_%{domain}_%{folds}'

def generate_command(classifier, representation, corpus, selection, domain=nil, folds=nil, layers=nil)
  cmd = BASE_CMD
  cmd_hash = {classifier: classifier, representation: representation, corpus: corpus, selection: selection}
  results = BASE_RESULTS
  results_hash = Hash.new('NA')
  results_hash[:classifier] = classifier
  results_hash[:corpus] = corpus
  results_hash[:representation] = representation
  results_hash[:selection] = selection

  if domain != nil
    cmd += '--word_vector_model_path ../resources/word_vectors/%{domain}.bin.gz '
    cmd += '--vector_domain %{domain} '
    cmd_hash[:domain] = domain
    results_hash[:domain] = domain
  end

  if folds != nil
    cmd += '--folds %{folds} '
    cmd_hash[:folds] = folds
    results_hash[:folds] = folds
  end

  if layers != nil
    cmd += '--layers %{layers} '
    cmd_hash[:layers] = layers.gsub('_', ' ')
    results_hash[:layers] = layers
  end

  cmd_hash[:results_base] = results % results_hash
  cmd % cmd_hash
end

CLASSIFIERS.product(REPRESENTATIONS, CORPORA, CANDIDATES_SELECTION, VECTOR_DOMAIN, LAYERS).each do
  |classifier, representation, corpus, selection, domain, layers|

  next if (not layers and classifier == 'mlp') or (layers and classifier != 'mlp')
  next if (representation == 'word_window' and not domain) or (representation != 'word_window' and domain)

  cmd = generate_command(classifier, representation, corpus, selection, domain, nil, layers)
  STDERR.puts cmd
  raise 'Error on command' unless system cmd

  cmd = generate_command(classifier, representation, corpus, selection, domain, 3, layers)
  STDERR.puts cmd
  raise 'Error on command' unless system cmd
end

