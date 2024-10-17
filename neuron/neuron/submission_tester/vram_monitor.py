import threading
import time
from concurrent.futures import Executor, Future

from neuron import Contest

POLL_RATE_SECONDS = 0.1


class VRamMonitor:
    _future: Future
    _contest: Contest
    _vram_usage: int = 0
    _stop_flag: threading.Event

    def __init__(self, contest: Contest, executor: Executor):
        self._contest = contest
        self._stop_flag = threading.Event()

        self._future = executor.submit(self.monitor)

    def monitor(self):
        while not self._stop_flag.is_set():
            vram = self._contest.get_vram_used()

            if self._vram_usage < vram:
                self._vram_usage = vram

            time.sleep(POLL_RATE_SECONDS)

    def complete(self) -> int:
        self._stop_flag.set()
        self._future.result()

        return self._vram_usage
