# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import os
import sys

from functools import partial
from multiprocessing import Pool
from tabulate import tabulate
from thesis.utils import find


def process_file(ifile, meta):
    print('Processing %s' % ifile, file=sys.stderr)
    basename = os.path.basename(ifile)
    ofile = os.path.join(args.output, basename)
    with open(ifile, 'r') as fin, open(ofile, 'w') as fout:
        sentence = []
        sentences = 0

        for line in fin:
            line = line.strip().split()

            if not line and sentence:
                print('META:%s    sentence:%05d    file:%s    words:%03d'
                      % (meta, sentences, basename, len(sentence)), file=fout)
                print(tabulate(sentence, tablefmt='plain'), end='\n\n', file=fout)
                sentence = []
                sentences += 1
            elif line:
                sentence.append(line)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('meta')
    parser.add_argument('--pattern', default='*')
    parser.add_argument('--workers', type=int, default=12)

    args = parser.parse_args()

    with Pool(args.workers) as p:
        p.map(partial(process_file, meta=args.meta), find(args.input, args.pattern))

