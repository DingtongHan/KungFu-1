import multiprocessing as mp
import os

from kungfu.loader import _load_clib
from ctypes import c_int

def run():
    clib = _load_clib('libkungfu')
    clib.kungfu_run_main()

def _send_monitor_signal(sig):
    clib = _load_clib('libkungfu')
    clib.kungfu_run_send_signal(c_int(sig))


def monitor_batch_begin():
    _send_monitor_signal(1)


def monitor_batch_end():
    _send_monitor_signal(2)


def monitor_train_end():
    _send_monitor_signal(3)


def monitor_epoch_end():
    _send_monitor_signal(4)


class _RunWorker(object):
    def __init__(self, f, np):
        self._np = np
        self._f = f

    def __call__(self, rank):
        from kungfu.python import _init_single_machine_multiple_process
        _init_single_machine_multiple_process(rank, self._np)
        self._f(rank)


def launch_multiprocess(f, np):
    child_env = os.environ.copy()
    child_env['KUNGFU_SINGLE_MACHINE_MULTIPROCESS'] = 'true'
    with mp.Pool(np, initargs=(child_env, )) as p:
        p.map(_RunWorker(f, np), range(np))
