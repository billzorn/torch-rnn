#!/usr/bin/env python

import re
import random
import json
import codecs
import numpy as np
import h5py

gutenberg_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'

license_block = '''<<THIS ELECTRONIC VERSION OF THE COMPLETE WORKS OF WILLIAM
SHAKESPEARE IS COPYRIGHT 1990-1993 BY WORLD LIBRARY, INC., AND IS
PROVIDED BY PROJECT GUTENBERG ETEXT OF ILLINOIS BENEDICTINE COLLEGE
WITH PERMISSION.  ELECTRONIC AND MACHINE READABLE COPIES MAY BE
DISTRIBUTED SO LONG AS SUCH COPIES (1) ARE FOR YOUR OR OTHERS
PERSONAL USE ONLY, AND (2) ARE NOT DISTRIBUTED OR USED
COMMERCIALLY.  PROHIBITED COMMERCIAL DISTRIBUTION INCLUDES BY ANY
SERVICE THAT CHARGES FOR DOWNLOAD TIME OR FOR MEMBERSHIP.>>'''

def gutenberg_fix_txt(s):
    title = '''1593

THE COMEDY OF ERRORS

by William Shakespeare'''
    orig = re.escape(title) + r'\s*' + re.escape(license_block)
    fixed = license_block + '\n\n\n\n' + title + '\n'
    return re.sub(orig, fixed, re.sub(r'\r\n', '\n', s))

# play separator
sep_1 = '\x1c'
# pseudo act separator
sep_2 = '\x1d'

def unpack(fname):
    print('unpacking {:s}'.format(fname))
    with codecs.open(fname, 'r', 'utf-8') as f:
        s = f.read()
        print('characters {:d}'.format(len(s)))
        fragments = s.split(license_block)
        print('fragments {:d}'.format(len(fragments)))
        spliced = ''.join(fragments)
        brackets = re.findall(r'[<>]', spliced)
        print('characters after fragmentation {:d}'.format(len(spliced)))

        lines = spliced.split('\n')
        print('lines {:d}'.format(len(lines)))

        info = fragments[0:2] + [fragments[-1]]
        sonnets = fragments[2].strip()
        poem = fragments[-2].strip()
        fragments = fragments[3:-2]

        plays = {}
        current_play = []
        current_name = ''

        for i, fragment in enumerate(fragments):
            f = fragment.strip()
            if re.match(r'[0123456789]+', f[:4]):
                if current_name != '':
                    plays[current_name] = current_play
                f = f[4:]
                current_name = f.strip().split('\n')[0]
                current_play = []
            current_play.append(f)

        if current_name != '':
            plays[current_name] = current_play
                

        print('found {:d} plays'.format(len(plays)))

        # for name in sorted(plays):
        #     print('{:s} ({:d} acts)'.format(name, len(plays[name])))
        #     for act in plays[name]:
        #         print('  {:s} ... {:s} ({:d})'.format(repr(act[:40]), repr(act[-40:]), len(act)))

        for name in sorted(plays):
            print('  {:s} ({:d} acts)'.format(name, len(plays[name])))

        return plays

def split(plays, seed = 109081843):
    random.seed(seed)
    print('dividing into train and val fracs')
    names = plays.keys()
    random.shuffle(names)
    half = len(names) // 2
    train = names[:half]
    val = names[half:]

    plays_train = {}
    plays_val = {}

    for name in train:
        plays_train[name] = plays[name]

    for name in val:
        acts = plays[name]
        val_idx = random.randint(0, len(acts) - 1)
        plays_val[name] = acts[val_idx:val_idx+1]
        plays_train[name] = acts[:val_idx] + acts[val_idx+1:]

    print('training')
    for name in sorted(plays_train):
        print('  {:s} ({:d} acts)'.format(name, len(plays_train[name])))

    print('val')
    for name in sorted(plays_val):
        print('  {:s} ({:d} acts)'.format(name, len(plays_val[name])))

    return plays_train, plays_val

def vocab(plays):
    tokens = {sep_1, sep_2}
    for play in plays:
        for act in plays[play]:
            for c in act:
                if not c in tokens:
                    tokens.add(c)
    
    token_to_idx = {tok:i+1 for i, tok in enumerate(sorted(tokens))}                    
    idx_to_token = {i+1:tok for i, tok in enumerate(sorted(tokens))}

    return token_to_idx, idx_to_token

def encode(train_fname, val_fname, plays, plays_train, plays_val, seed = 986969689423, h5_fname = None):
    random.seed(seed)

    train_names = plays_train.keys()
    random.shuffle(train_names)
    train_corpus = sep_1.join([sep_2.join(plays_train[name]) for name in train_names])

    val_names = plays_val.keys()
    random.shuffle(val_names)
    val_corpus = sep_1.join([sep_2.join(plays_val[name]) for name in val_names])

    print('writing training data to {:s}'.format(train_fname))
    with open(train_fname, 'wt') as f:
        f.write(train_corpus)

    print('writing validation data to {:s}'.format(val_fname))
    with open(val_fname, 'wt') as f:
        f.write(val_corpus)

    if h5_fname is None:
        return

    # else encode the corpus as an h5 file as well

    train_size = len(train_corpus)
    val_size = len(val_corpus)
    test_size = 0

    token_to_idx, idx_to_token = vocab(plays)
    dtype = np.uint8
    if len(token_to_idx) > 255:
        dtype = np.uint32

    train_data = np.zeros(train_size, dtype=dtype)
    val_data = np.zeros(val_size, dtype=dtype)
    test_data = np.zeros(test_size, dtype=dtype)

    for i, c in enumerate(train_corpus):
        train_data[i] = token_to_idx[c]

    for i, c in enumerate(val_corpus):
        val_data[i] = token_to_idx[c]

    with h5py.File(h5_fname, 'w') as f:
        f.create_dataset('train', data=train_data)
        f.create_dataset('val', data=val_data)
        f.create_dataset('test', data=test_data)


if __name__ == '__main__':
    import os
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('fds', type=int, nargs='*',
                        help='file descriptors to write streams to')
    parser.add_argument('-f', '--fname', required=True,
                        help='file to read shakespeare from')
    parser.add_argument('-n', '--block_size', type=int, default=10000,
                        help='number of characters each stream should read/write at a time')
    parser.add_argument('--train',
                        help='file to write training corpus to. specify with --val')
    parser.add_argument('--val',
                        help='file to write validation corpus to. specify with --train')
    parser.add_argument('--json',
                        help='json file to optionally write the vocab to')
    parser.add_argument('--h5',
                        help='h5 file to optionally write the corpus to')

    args = parser.parse_args()

    # download from project gutenberg if the file doesn't already exist
    if not os.path.exists(args.fname):
        import urllib2
        webf = urllib2.urlopen(gutenberg_url)
        s = gutenberg_fix_txt(webf.read())
        with open(args.fname, 'wt') as f:
            f.write(s)

    plays = unpack(args.fname)

    if args.json:
        token_to_idx, idx_to_token = vocab(plays)
        json_data = {'token_to_idx':token_to_idx, 'idx_to_token':idx_to_token}
        print('writing vocabulary ({:d}) to {:s}'.format(len(token_to_idx), args.json))
        with open(args.json, 'w') as f:
            json.dump(json_data, f)

    plays_train, plays_val = split(plays)

    if args.train and args.val:
        encode(args.train, args.val, plays, plays_train, plays_val, h5_fname=args.h5)
    else:
        print('will stream')

