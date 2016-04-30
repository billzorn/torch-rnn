#!/usr/bin/env python

import os
import threading
import time
import signal
import traceback
import psutil

# correctly setting up a stream that won't get orphaned and left clutting the operating
# system proceeds in 3 parts:
#   1) invoke install_suicide_handlers() to ensure correct behavior on interrupt
#   2) get threads by invoking spawn_stream_threads
#   3) invoke wait_and_kill_self_noreturn(threads)
# or, use the handy wrapper that does it for you

def spawn_stream_threads(fds, runthread, mkargs):
    threads = []
    for i, fd in enumerate(fds):
        stream_thread = threading.Thread(target=runthread, args=mkargs(i, fd))
        stream_thread.daemon = True
        stream_thread.start()
        threads.append(stream_thread)
    return threads

def force_kill_self_noreturn():
    # We have a strange issue here, which is that our threads will refuse to die
    # to a normal exit() or sys.exit() because they're all blocked in write() calls
    # on full pipes; the simplest workaround seems to be to ask the OS to terminate us.
    # This works, but...
    #os.kill(os.getpid(), signal.SIGTERM)
    # psutil might have useful features like checking if the pid has been reused before killing it.
    # Also we might have child processes like l2e luajits to think about.
    me = psutil.Process(os.getpid())
    for child in me.children(recursive=True):
        child.terminate()
    me.terminate()

def handler_kill_self(signum, frame):
    traceback.print_stack(frame)
    print('caught signal {:d} - streamer sending SIGTERM to self'.format(signum))
    force_kill_self_noreturn()

def install_suicide_handlers():
    for sig in [signal.SIGHUP, signal.SIGINT, signal.SIGQUIT]:
        signal.signal(sig, handler_kill_self)

def wait_and_kill_self_noreturn(threads):
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
    force_kill_self_noreturn()

def streaming_noreturn(fds, write_stream, mkargs):
    install_suicide_handlers()
    threads = spawn_stream_threads(fds, write_stream, mkargs)
    wait_and_kill_self_noreturn(threads)
    assert False, 'should not return from streaming'


def main(args):
    fds = args.fds
    fname = args.fname
    block_size =  args.block_size

    flen = os.path.getsize(fname)
    offset = max(flen // len(fds), 1)

    def write_stream(fd, offset = 0):
        with open('/proc/self/fd/'+str(fd), 'wt') as f_stream:
            with open(fname, 'rt') as f_in:
                f_in.seek(offset)
                while True:
                    text = f_in.read(block_size)
                    f_stream.write(text)
                    if len(text) < block_size:
                        f_in.seek(0)

    def mkargs(i, fd):
        return (fd, (i*offset)%flen)

    streaming_noreturn(fds, write_stream, mkargs)
    

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
