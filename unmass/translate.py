# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | \
#     python translate.py --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#

import os
import io
import sys
import argparse
import torch
import io

from unmass.utils import AttrDict
from unmass.utils import bool_flag
from unmass.data.dictionary import Dictionary
from unmass.model.transformer import TransformerModel

import logging as logger

from unmass.fp16 import network_to_half

logger.basicConfig(level=logger.INFO)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters

    parser = argparse.ArgumentParser(description="Translate sentences")
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore')
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    # main parameters
    parser.add_argument("--fp16", type=bool_flag, default=False, help="Run model with float16")
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("-m", "--model", type=str, required=True, help="Model path")
    parser.add_argument("-i", "--input", default=stdin,
                        type=argparse.FileType('r', encoding='utf-8', errors='ignore'),
                        help="Input path. Default: STDIN")
    parser.add_argument("-o", "--output", default=stdout,
                        type=argparse.FileType('w', encoding='utf-8', errors='ignore'),
                        help="Output path: Default: STDOUT")

    parser.add_argument("-bm", "--beam", type=int, default=0.6, help="Beam size")
    parser.add_argument("-lp", "--length_penalty", type=float, default=1, help="length penalty")

    # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # source language / target language
    parser.add_argument("-sl", "--src_lang", type=str, required=True, help="Source language")
    parser.add_argument("-tl", "--tgt_lang", type=str, required=True, help="Target language")

    return parser


def unwrap(state):

    if all(key.startswith('module.') for key in state):
        logger.info("unwrapping module")
        off = len("module.")
        state = {key[off:]: val for key, val in state.items()}

    return state


def main(params=None):
    # generate parser / parse parameters
    parser = get_parser()
    params = params or parser.parse_args()
    reloaded = torch.load(params.model, map_location=device)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # update dictionary parameters
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True)
    decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True)
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    encoder.load_state_dict(unwrap(reloaded['encoder']))
    decoder.load_state_dict(unwrap(reloaded['decoder']))
    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]

    # float16
    if params.fp16:
        assert torch.backends.cudnn.enabled
        encoder = network_to_half(encoder)
        decoder = network_to_half(decoder)

    # read sentences from stdin
    src_sent = []
    for i, line in enumerate(params.input):
        line = line.strip()
        assert line, f'Found an empty line at line number: {i}. Please remove it.'
        src_sent.append(line)
    logger.info(f"Read {len(src_sent)} sentences from input. Translating ...")

    out = params.output

    for i in range(0, len(src_sent), params.batch_size):

        # prepare batch
        word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                    for s in src_sent[i:i + params.batch_size]]
        lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index)
        batch[0] = params.eos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = params.eos_index
        langs = batch.clone().fill_(params.src_id).to(device)
        batch = batch.to(device)
        lengths = lengths.to(device)
        # encode source batch and translate it
        encoded = encoder('fwd', x=batch, lengths=lengths, langs=langs,
                          causal=False)
        encoded = encoded.transpose(0, 1)
        if params.beam == 1:
            decoded, dec_lengths = decoder.generate(encoded, lengths, params.tgt_id,
                                                    max_len=int(1.5 * lengths.max().item() + 10))
        else:
            decoded, dec_lengths = decoder.generate_beam(
                encoded, lengths, params.tgt_id, beam_size=params.beam,
                length_penalty=params.length_penalty,
                early_stopping=False,
                max_len=int(1.5 * lengths.max().item() + 10))

        # convert sentences to words
        for j in range(decoded.size(1)):
            # remove delimiters
            sent = decoded[:, j]
            delimiters = (sent == params.eos_index).nonzero().view(-1)
            assert len(delimiters) >= 1 and delimiters[0].item() == 0
            sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

            # output translation
            source = src_sent[i + j].strip()
            target = " ".join([dico[sent[k].item()] for k in range(len(sent))])
            logger.info("%i / %i: %s -> %s" % (i + j, len(src_sent), source, target))
            out.write(target + "\n")


def cli():
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.model)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang

    # translate
    with torch.no_grad():
        main(params)


if __name__ == '__main__':
    cli()
