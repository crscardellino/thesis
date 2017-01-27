#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import sys

from corpora.processors.ancora import AncoraCorpusReader
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("corpus",
                        help="Ancora Corpus root directory")
    parser.add_argument("--output",
                        default=None,
                        help="Output file to write (defaults to STDOUT)")
    args = parser.parse_args()

    reader = AncoraCorpusReader(args.corpus)

    output = sys.stdout if args.output is None else open(args.output, "w")

    print("Parsing Ancora corpus", file=sys.stderr)
    for sidx, sentence in tqdm(enumerate(reader.sentences, start=1)):
        print("META:ancora\tsentence:%05d" % sidx, file=output)
        for word in sentence:
            print("\t".join(word), file=output)
        print("", file=output)

    if args.output is not None:
        output.close()

    print("Ancora corpus parsed", file=sys.stderr)
