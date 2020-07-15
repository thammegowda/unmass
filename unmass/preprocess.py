#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#


"""
Example: python data/vocab.txt data/train.txt
vocab.txt: 1stline=word, 2ndline=count
"""

import os
import sys

from unmass.logger import create_logger
from unmass.data.dictionary import Dictionary


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('vocab', help='Path to Vocabulary')
    p.add_argument('text', help='Path to input text file')
    p.add_argument("bin", nargs='?', help='Path to store the binarized output.'
                                             ' default = <text_arg>.pth')
    args = p.parse_args()
    assert os.path.isfile(args.vocab)
    assert os.path.isfile(args.text)
    if not args.bin:
        args.bin = args.text + '.pth'
    return args

if __name__ == '__main__':

    logger = create_logger(None, 0)
    args = parse_args()

    dico = Dictionary.read_vocab(args.vocab)
    logger.info("")

    data = Dictionary.index_data(args.text, args.bin, dico)
    logger.info("%i words (%i unique) in %i sentences." % (
        len(data['sentences']) - len(data['positions']),
        len(data['dico']),
        len(data['positions'])
    ))
    if len(data['unk_words']) > 0:
        logger.info("%i unknown words (%i unique), covering %.2f%% of the data." % (
            sum(data['unk_words'].values()),
            len(data['unk_words']),
            sum(data['unk_words'].values()) * 100. / (len(data['sentences']) - len(data['positions']))
        ))
        if len(data['unk_words']) < 30:
            for w, c in sorted(data['unk_words'].items(), key=lambda x: x[1])[::-1]:
                logger.info("%s: %i" % (w, c))
