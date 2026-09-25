"""Non-networking STH-Brain telemetry producer. All tensors are detached before logging."""
from pathlib import Path
import json
import math
import queue
import threading
import time
import warnings



def finite_scalars(row):
    """Flatten detached diagnostic trees; preserve existing scalar metric keys.

    Lists use zero-based indices (e.g. subject_counts/0). Tensors and non-finite
    numbers are deliberately excluded so the writer never owns a training graph.
    """
    result={}
    def visit(key,value):
        if isinstance(value,dict):
            for name,child in value.items():visit(f'{key}/{name}',child)
        elif isinstance(value,(list,tuple)):
            for index,child in enumerate(value):visit(f'{key}/{index}',child)
        elif isinstance(value,(bool,int,float,str)) and not (isinstance(value,float) and not math.isfinite(value)):
            result[key]=value
    for key,value in row.items():visit(str(key),value)
    return result


class Journal:
    """Bounded nonblocking producer; local disk and W&B never run on train thread."""
    def __init__(self, run_dir):
        self.queue = queue.Queue(maxsize=4096)
        self.dropped = 0
        self.error = ''
        self.path = Path(run_dir) / 'telemetry/events.jsonl'
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def log(self, key, payload):
        try:
            self.queue.put_nowait({'key': key, 'payload': finite_scalars(payload),
                                   'observed_at_unix': time.time()})
        except queue.Full:
            self.dropped += 1
            if self.dropped == 1:
                warnings.warn('Telemetry queue full: metrics dropped; training continues')

    def _write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', buffering=1) as handle:
                while True:
                    event = self.queue.get()
                    try:
                        if event is None:
                            return
                        handle.write(json.dumps(event, allow_nan=False) + '\n')
                    finally:
                        self.queue.task_done()
        except Exception as exc:
            self.error = type(exc).__name__
            warnings.warn(f'Telemetry local writer failed ({self.error}); training continues')

    def close(self):
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            self.dropped += 1
        self.thread.join(timeout=2)
        if self.thread.is_alive() or self.dropped or self.error:
            warnings.warn(f'Telemetry incomplete: dropped={self.dropped}, error={self.error}')




