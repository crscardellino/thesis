#!/usr/bin/env ruby

Dir.chdir '..'

CLASSIFIERS = %w(svm mlp)
REPRESENTATIONS = %w(hashed word_window)
CORPORA = %w(sensem)
LAYERS = [nil] + %w(100 250_100)
VECTOR_DOMAIN = [nil] + %w(journal)
ACCEPTANCE_THRESHOLD = 0.9
VALIDATION_RATIO = 0.2
ERROR_SIGMA = 0.1
UNLABELED_DATA_LIMIT = 1000

LANGUAGE = {sensem: 'spanish', semeval: 'english'}

BASE_CMD = 'python -W ignore -m thesis.classification.selflearning '
BASE_CMD.concat('../resources/%{representation}/%{corpus} ')
BASE_CMD.concat('../resources/%{representation}/unlabeled_%{language} ')
BASE_CMD.concat('../results/selflearning/%{results_base} ')
BASE_CMD.concat('--classifier %{classifier} ')
BASE_CMD.concat("--acceptance_threshold #{ACCEPTANCE_THRESHOLD} ")
BASE_CMD.concat("--error_sigma #{ERROR_SIGMA} ")
BASE_CMD.concat("--unlabeled_data_limit #{UNLABELED_DATA_LIMIT} ")
BASE_CMD.concat("--validation_ratio #{VALIDATION_RATIO} ")
BASE_CMD.concat('--min_count 2 ')
BASE_CMD.concat('--corpus_name %{corpus} ')
BASE_CMD.concat('--representation %{representation} ')

BASE_RESULTS = '%{corpus}_%{classifier}_%{representation}_%{layers}_%{domain}'

def generate_command(classifier, representation, corpus, language, domain=nil, layers=nil)
  cmd = BASE_CMD
  cmd_hash = {classifier: classifier, representation: representation, corpus: corpus, language: language}
  results = BASE_RESULTS
  results_hash = Hash.new('NA')
  results_hash[:classifier] = classifier
  results_hash[:corpus] = corpus
  results_hash[:representation] = representation

  if domain != nil
    cmd += '--word_vector_model_path ../resources/word_vectors/%{domain}.bin.gz '
    cmd += '--vector_domain %{domain} '
    cmd_hash[:domain] = domain
    results_hash[:domain] = domain
  end

  if layers != nil
    cmd += '--layers %{layers} '
    cmd_hash[:layers] = layers.gsub('_', ' ')
    results_hash[:layers] = layers
  end

  cmd_hash[:results_base] = results % results_hash
  cmd % cmd_hash
end

CLASSIFIERS.product(REPRESENTATIONS, CORPORA, VECTOR_DOMAIN, LAYERS).each do
  |classifier, representation, corpus, domain, layers|

  next if (not layers and classifier == 'mlp') or (layers and classifier != 'mlp')
  next if (representation == 'word_window' and not domain) or (representation != 'word_window' and domain)

  cmd = generate_command(classifier, representation, corpus, LANGUAGE[corpus.to_sym], domain, layers)
  STDERR.puts cmd
  raise 'Error on command' unless system cmd
end

