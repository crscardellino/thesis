#!/usr/bin/env python

from __future__ import print_function, unicode_literals

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--corpus_name')
    parser.add_argument('--representation')
    parser.add_argument('--classifier')
    parser.add_argument('--splits', type=int)
    parser.add_argument('--folds', type=int)
    parser.add_argument('--vector_domain', default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=list())
    parser.add_argument('--min_count', type=int, default=2)

    args = parser.parse_args()
    args.layers = [args.layers] if isinstance(args.layers, int) else args.layers

    dataset = "../resources/%s/%s" % (args.representation, args.corpus_name)
    word_vectors_model = "../resources/word_vectors/%s.bin.gz" % args.vector_domain if args.vector_domain is not None else ""

    print("#!/usr/bin/env bash")
    print("set -e")
    print("cd ..", end="\n\n")

    print("python -m thesis.classification \\")
    print("    %s/train_dataset.npz \\" % dataset)
    print("    %s/test_dataset.npz \\" % dataset)

    name_layers = '_'.join(str(l) for l in args.layers)

    if args.layers != [] and args.vector_domain is not None:
        print("    %s/%s_%s_%s_%s_%s.csv \\" % (args.results, args.corpus_name, args.classifier, args.representation, name_layers, args.vector_domain))
    elif args.layers != []:
        print("    %s/%s_%s_%s_%s_%s.csv \\" % (args.results, args.corpus_name, args.classifier, args.representation, name_layers, 'NA'))
    elif args.vector_domain is not None:
        print("    %s/%s_%s_%s_%s_%s.csv \\" % (args.results, args.corpus_name, args.classifier, args.representation, 'NA', args.vector_domain))
    else:
        print("    %s/%s_%s_%s_%s_%s.csv \\" % (args.results, args.corpus_name, args.classifier, args.representation, 'NA', 'NA'))

    print("    --classifier %s \\" % args.classifier)
    if args.layers != []:
        print("    --layers %s \\" % name_layers.replace('_', ' '))
    
    if args.vector_domain is not None:
        print("    --word_vectors_model_path %s \\" % word_vectors_model)
        print("    --vector_domain %s \\" % args.vector_domain)
    print("    --min_count %d \\" % args.min_count)
    print("    --corpus_name %s \\" % args.corpus_name)
    print("    --representation %s" % args.representation, end='\n\n')

    print("python -m thesis.classification \\")
    print("    %s/train_dataset.npz \\" % dataset)
    print("    %s/test_dataset.npz \\" % dataset)

    name_layers = '_'.join(str(l) for l in args.layers)

    if args.layers != [] and args.vector_domain is not None:
        print("    %s/%s_%s_%s_%s_%s_%d_%d.csv \\" % (args.results, args.corpus_name,
            args.classifier, args.representation, name_layers, args.vector_domain, args.splits, args.folds))
    elif args.layers != []:
        print("    %s/%s_%s_%s_%s_%s_%d_%d.csv \\" % (args.results, args.corpus_name,
            args.classifier, args.representation, name_layers, 'NA', args.splits, args.folds))
    elif args.vector_domain is not None:
        print("    %s/%s_%s_%s_%s_%s_%d_%d.csv \\" % (args.results, args.corpus_name,
            args.classifier, args.representation, 'NA', args.vector_domain, args.splits, args.folds))
    else:
        print("    %s/%s_%s_%s_%s_%s_%d_%d.csv \\" % (args.results, args.corpus_name,
            args.classifier, args.representation, 'NA', 'NA', args.splits, args.folds))

    print("    --classifier %s \\" % args.classifier)
    if args.layers != []:
        print("    --layers %s \\" % name_layers.replace('_', ' '))

    print("    --splits %d \\" % args.splits)
    print("    --folds %d \\" % args.folds)
    
    if args.vector_domain is not None:
        print("    --word_vectors_model_path %s \\" % word_vectors_model)
        print("    --vector_domain %s \\" % args.vector_domain)
    print("    --min_count %d \\" % args.min_count)
    print("    --corpus_name %s \\" % args.corpus_name)
    print("    --representation %s" % args.representation)
