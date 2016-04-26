#!/usr/bin/env python

import os
import threading
import time
import signal

def write_stream(fd, fname, offset = 0, block_size = 10000):
    with open('/proc/self/fd/'+str(fd), 'wt') as f_stream:
        with open(fname, 'rt') as f_in:
            f_in.seek(offset)
            while True:
                text = f_in.read(block_size)
                f_stream.write(text)
                if len(text) < block_size:
                    f_in.seek(0)

def spawn_threads(fds, fname, block_size = 10000):
    flen = os.path.getsize(fname)
    offset = max(flen // len(fds), 1)

    threads = []
    for i, fd in enumerate(args.fds):
        stream_thread = threading.Thread(target=write_stream,
                                         args=(fd, fname, (i*offset)%flen, block_size))
        stream_thread.daemon = True
        stream_thread.start()
        threads.append(stream_thread)
    return threads

def main(args):
    fds = args.fds
    fname = args.fname
    block_size =  args.block_size
    
    # wait for infinite loop
    threads = spawn_threads(fds, fname, block_size)
    running = True
    while running:
        running = False
        for thread in threads:
            if thread.is_alive():
                running = True
        if(os.getppid() <= 1):
            # exit if parent process died (and we were reparented to init)
            break
        time.sleep(1)
    # We have a strange issue here, which is that our threads will refuse to die
    # to a normal exit() or sys.exit() because they're all blocked in write() calls
    # on full pipes; the simples workaround seems to be to ask the OS to terminate us
    os.kill(os.getpid(), signal.SIGTERM)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('fds', type=int, nargs='+',
                        help='file descriptors to write streams to')
    parser.add_argument('-f', '--fname', required=True,
                        help='file to read streams from')
    parser.add_argument('-n', '--block_size', type=int, default=10000,
                        help='number of characters each stream should read/write at a time')
    args = parser.parse_args()

    main(args)
