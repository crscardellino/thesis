#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import sys

from corpora.processors.sensem import SenSemCorpusReader
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("corpus",
                        help="SenSem Corpus root directory")
    parser.add_argument("--output",
                        default=None,
                        help="Output file to write (defaults to STDOUT)")
    args = parser.parse_args()

    reader = SenSemCorpusReader(args.corpus)

    output = sys.stdout if args.output is None else open(args.output, "w")

    print("Parsing SenSem corpus", file=sys.stderr)
    for sidx, (metadata, sentence) in tqdm(enumerate(reader.sentences, start=1)):
        print("META:sensem\tsentence:%05d\t%s" % (sidx, "\t".join(map(lambda d: ":".join(d), metadata))), file=output)
        for word in sentence:
            print("\t".join(word), file=output)
        print("", file=output)

    if args.output is not None:
        output.close()

    print("SenSem corpus parsed", file=sys.stderr)
