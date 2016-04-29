#!/usr/bin/env python

import os
import sys
import subprocess

def get_last_cp_batch(dirname, cpprefix, cpsuffix):
    last_cp_batch = 0
    last_cp_path = ''
    for cp in os.listdir(dirname):
        if cp[:len(cpprefix)] == cpprefix and cp[-len(cpsuffix):] == cpsuffix:
            try:                
                underscore_batchnum = cp[len(cpprefix):-len(cpsuffix)]
                batchnum = int(underscore_batchnum[1:])
                if batchnum > last_cp_batch:
                    last_cp_batch = batchnum
                    last_cp_path = os.path.join(dirname, cp)
            except Exception as e:
                print('invalid CP name: {:s}'.format(cp))
                print(e)
    return last_cp_path, last_cp_batch

def extract_args(argv, keys):
    key_args = {}
    other_args = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in keys and i < len(argv) - 1:
            assert(not arg in key_args)
            key_args[arg] = argv[i+1]
            i += 2
        else:
            other_args.append(arg)
            i += 1
    return key_args, other_args

def pprint_cmdline(cmd):
    sys.stdout.write(' ')
    for s in cmd:
        if s[:1] == '-':
            sys.stdout.write('\n    ')
        else:
            sys.stdout.write(' ')
        sys.stdout.write(repr(s)[1:-1])
    sys.stdout.write('\n')

def main(args):
    cp_dir, cp_prefix = os.path.split(args.checkpoint_name)
    cp_suffix = '.t7'
    checkpoint_every = args.checkpoint_every
    max_batches = args.max_batches
    
    # Currently the script throws away -init_from and -reset_training_history.
    init_from = ''
    reset_training_history = 0
    # Also -reset_training_position.
    reset_training_position = 0

    # Not supported as an argument, but could be
    max_retries = 3

    cmd = args.args + ['-checkpoint_name', args.checkpoint_name,
                       '-checkpoint_every', str(checkpoint_every),
                       '-max_batches', str(max_batches),]

    # make the checkpoint directoy, if it doesn't exist
    if not os.path.exists(cp_dir):
        os.path.makedirs(cp_dir)

    cp, batch = get_last_cp_batch(cp_dir, cp_prefix, cp_suffix)

    print('--------')
    print('{:s}: restarting training until {:d} batches'.format(sys.argv[0], max_batches))
    print('  or {:d} consecutive failures to make progress.'.format(max_retries))
    if cp != '':
        print('Found existing checkpoint {:s}, batch {:d}'.format(cp, batch))
    print('Command:')
    pprint_cmdline(cmd)
    print('--------')

    print('\nPress enter to continue, or C-c to interrupt')
    sys.stdin.readline()

    first = True
    retries = 0
    while batch < max_batches:
        this_cmd = [s for s in cmd]
        if cp != '':
            this_cmd += ['-init_from', cp,
                         '-reset_training_history', str(reset_training_history),
                         '-reset_training_position', str(reset_training_position),]

        print('\n--------')
        print('starting from {:s}, batch {:d}'.format(cp, batch))
        print(' '.join(this_cmd))
        print('--------\n')

        status = subprocess.call(this_cmd)

        cp, newbatch = get_last_cp_batch(cp_dir, cp_prefix, cp_suffix)

        if not first and status == 0 and newbatch % checkpoint_every != 0:
            print('\n\nDetected end of training, exiting.')
            break

        if newbatch <= batch:
            if retries >= max_retries:
                print('\n\nRestarted {:d} times without progress, aborting'.format(retries))
                break
            retries += 1
            print('\n\nFailed to create a new checkpoint: retrying ({:d})'.format(retries))
        else:
            retries = 0
            first = False
            batch = newbatch

    print('\n\nDone, reached target at {:s}, batch {:d}'.format(cp, batch))


if __name__ == '__main__':
    keys = {
        '-checkpoint_name' : str,
        '-checkpoint_every' : int,
        '-max_batches' : int,
        '-init_from' : str,
        '-reset_training_history' : int,
        '-reset_training_position' : int,
    }
    key_args, other_args = extract_args(sys.argv, keys)

    # shift some keywords we care about to the beginning of the argument list
    argv2 = []
    for key in key_args:
        argv2 += [key, key_args[key]]
    argv2 += other_args[1:]

    # then parse them with argparse so we have a nice interface
    import argparse
    parser = argparse.ArgumentParser()
    for key in keys:
        parser.add_argument(key, type=keys[key])
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv2)

    main(args)
