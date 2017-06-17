#!/usr/bin/env ruby

Dir.chdir '..'

REPRESENTATIONS = %w(hashed word_window)
CORPORA = %w(sensem)
LAYERS = %w(100 250_100 100_100_100 250_100_100 250_250_100 500_250_100 500_250_250_100)
EPOCHS = %w(25 50)
NOISE = %w(0.2 0.3 0.5)
VECTOR_DOMAIN = [nil] + %w(journal)
ACCEPTANCE_THRESHOLD = 0.9
VALIDATION_RATIO = 0.2
ERROR_SIGMA = 0.1
UNLABELED_DATA_LIMIT = 1000

LANGUAGE = {sensem: 'spanish', semeval: 'english'}

BASE_CMD = 'python -W ignore -m thesis.classification.ladder '
BASE_CMD.concat('../resources/%{representation}/%{corpus} ')
BASE_CMD.concat('../resources/%{representation}/unlabeled_%{language} ')
BASE_CMD.concat('../results/ladder/%{results_base} ')
BASE_CMD.concat("--acceptance_threshold #{ACCEPTANCE_THRESHOLD} ")
BASE_CMD.concat("--error_sigma #{ERROR_SIGMA} ")
BASE_CMD.concat("--unlabeled_data_limit #{UNLABELED_DATA_LIMIT} ")
BASE_CMD.concat("--validation_ratio #{VALIDATION_RATIO} ")
BASE_CMD.concat('--min_count 2 ')
BASE_CMD.concat('--epochs %{epochs} ')
BASE_CMD.concat('--noise_std %{noise} ')
BASE_CMD.concat('--corpus_name %{corpus} ')
BASE_CMD.concat('--representation %{representation} ')

BASE_RESULTS = '%{corpus}_%{representation}_%{layers}_%{domain}_%{epochs}_%{noise}'

def generate_command(representation, corpus, language, layers, epochs, noise, domain=nil)
  cmd = BASE_CMD
  cmd_hash = {representation: representation, corpus: corpus, language: language, epochs: epochs, noise: noise}
  results = BASE_RESULTS
  results_hash = Hash.new('NA')
  results_hash[:corpus] = corpus
  results_hash[:representation] = representation
  results_hash[:epochs] = epochs
  results_hash[:noise] = noise

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

    cmd += '--denoising_cost %{dc}'
    dc = []
    layers.split('_').each_with_index { |l, i| dc.push(if i == 0 then 1000.0 elsif i == 1 then 10.0 else 0.10 end) }
    cmd_hash[:dc] = dc.join(' ')
  end

  cmd_hash[:results_base] = results % results_hash
  cmd % cmd_hash
end

REPRESENTATIONS.product(CORPORA, VECTOR_DOMAIN, LAYERS, EPOCHS, NOISE).each do
  |representation, corpus, domain, layers, epochs, noise|

  next if (representation == 'word_window' and not domain) or (representation != 'word_window' and domain)

  cmd = generate_command(representation, corpus, LANGUAGE[corpus.to_sym], layers, epochs, noise, domain)
  STDERR.puts cmd
  raise 'Error on command' unless system cmd
end

