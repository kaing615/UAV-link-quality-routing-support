"""Real-process stdout drain and failure propagation at bridge shutdown."""
import subprocess
import sys
import threading

import pytest

from src.control.bridge import SimulatorBridge,ProtocolError
from src.control.prevention.config import PreventionConfig


def test_close_drains_delayed_reader_before_closing_pipe(tmp_path,monkeypatch):
    errors=[]
    monkeypatch.setattr(threading,'excepthook',lambda args:errors.append(args.exc_value))
    gate=threading.Event()
    class DelayedReaderBridge(SimulatorBridge):
        def _read_stdout(self):
            assert gate.wait(timeout=5)
            super()._read_stdout()
    expected=''.join(f'line {i}\n' for i in range(128))
    bridge=DelayedReaderBridge([sys.executable,'-c',"import sys;sys.stdout.write(''.join(f'line {i}\\n' for i in range(128)));sys.stdout.flush()"],
                              PreventionConfig(),tmp_path/'stderr.log',stdout_path=tmp_path/'stdout.log')
    # A real exited child with unread pipe data. Delay reader until close has
    # started, deterministically exposing close-before-reader ordering.
    assert bridge.process.wait(timeout=5)==0
    release=threading.Timer(.05,gate.set);release.start()
    try: bridge.close()
    finally: gate.set();release.join();bridge._reader.join(timeout=5)
    assert not errors,errors
    assert not bridge._reader.is_alive()
    assert (tmp_path/'stdout.log').read_text()==expected
    assert bridge._lines.qsize()==129  # 128 lines then exactly one EOF marker.
    bridge.close()  # idempotent after complete drain.


def test_reader_write_error_reaches_close(tmp_path):
    gate=threading.Event()
    class BrokenLog:
        def write(self,line): raise OSError('injected stdout log write failure')
        def flush(self):pass
        def close(self):pass
    class FailingReaderBridge(SimulatorBridge):
        def _read_stdout(self):
            assert gate.wait(timeout=5)
            super()._read_stdout()
    bridge=FailingReaderBridge([sys.executable,'-c',"print('one line')"],PreventionConfig(),tmp_path/'stderr.log')
    bridge._stdout_log=BrokenLog()
    gate.set()
    bridge.process.wait(timeout=5)
    with pytest.raises(ProtocolError,match='stdout.*injected stdout log write failure'):
        bridge.close()
    assert not bridge._reader.is_alive()
