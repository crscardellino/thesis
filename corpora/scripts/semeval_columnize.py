#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import sys

from corpora.processors import SemevalTrainCorpusReader, SemevalTestCorpusReader
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("corpus",
                        help="Semeval Corpus root directory")
    parser.add_argument("--output",
                        default=None,
                        help="Output file to write (defaults to STDOUT)")
    args = parser.parse_args()

    train_reader = SemevalTrainCorpusReader(args.corpus)

    test_reader = SemevalTestCorpusReader(args.corpus)

    output = sys.stdout if args.output is None else open(args.output, "w")

    print("Parsing Semeval corpus", file=sys.stderr)
    for sidx, (metadata, sentence) in tqdm(enumerate(train_reader.sentences, start=1)):
        print("META:semeval\tsentence:%05d\tcorpus:train\t%s" %
              (sidx, "\t".join(map(lambda d: ":".join(d), metadata))), file=output)
        for word in sentence:
            print("\t".join(word), file=output)
        print("", file=output)

    for sidx, (metadata, sentence) in tqdm(enumerate(test_reader.sentences, start=1)):
        print("META:semeval\tsentence:%05d\tcorpus:test\t%s" %
              (sidx, "\t".join(map(lambda d: ":".join(d), metadata))), file=output)
        for word in sentence:
            print("\t".join(word), file=output)
        print("", file=output)

    if args.output is not None:
        output.close()

    print("SenSem corpus parsed", file=sys.stderr)
