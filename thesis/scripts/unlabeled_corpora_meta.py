# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import os

from tabulate import tabulate
from thesis.utils import find
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('meta')
    parser.add_argument('--pattern', default='*')

    args = parser.parse_args()

    files = list(find(args.input, args.pattern))

    for ifile in tqdm(files):
        basename = os.path.basename(ifile)
        ofile = os.path.join(args.output, basename)
        with open(ifile, 'r') as fin, open(ofile, 'w') as fout:
            sentence = []
            sentences = 0

            for line in ifile:
                line = line.strip().split()

                if not line and sentence:
                    print('META:%s    sentence:%05d    file:%s    words:%03d'
                          % (args.meta, sentences, basename, len(sentence)))
                    print(tabulate(sentence, tablefmt='plain'), end='\n\n', file=fout)
                    sentence = []
                    sentences += 1
                elif line:
                    sentence.append(line)
