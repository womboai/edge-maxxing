from threading import Thread, Event
from time import sleep

from .. import Contest

SAMPLE_RATE_MS = 10

class VRamMonitor:
    _contest: Contest
    _thread: Thread
    _stop_flag: Event
    _vram_usage: int = 0

    def __init__(self, contest: Contest):
        self._contest = contest
        self._stop_flag = Event()

        self._thread = Thread(target=self._monitor)
        self._thread.start()

    def _monitor(self):
        while not self._stop_flag.is_set():
            self._vram_usage = max(self._vram_usage, self._contest.device.get_vram_used())
            sleep(SAMPLE_RATE_MS / 1000)

    def complete(self) -> int:
        self._stop_flag.set()
        self._thread.join()

        return self._vram_usage
