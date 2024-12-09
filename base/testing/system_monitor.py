from dataclasses import dataclass
from threading import Thread, Event

import psutil

from base.contest import Contest

SAMPLE_RATE_MS = 100

@dataclass
class SystemResults:
    vram_usage: int
    ram_usage: int
    cpu_usage: float

class SystemMonitor:
    _contest: Contest
    _thread: Thread
    _stop_flag: Event = Event()

    _process: psutil.Process = psutil.Process()
    _vram_usage: int = 0
    _ram_usage: int = 0
    _cpu_usage: float = 0

    def __init__(self, contest: Contest):
        self._contest = contest

        self._thread = Thread(target=self._monitor)
        self._thread.start()

    def _monitor(self):
        while not self._stop_flag.is_set():
            self._vram_usage = max(self._vram_usage, self._contest.device.get_vram_used())
            self._ram_usage = max(self._ram_usage, self._process.memory_info().rss)
            self._cpu_usage = max(self._cpu_usage, self._process.cpu_percent())

            self._stop_flag.wait(SAMPLE_RATE_MS / 1000)

    def complete(self) -> SystemResults:
        self._stop_flag.set()
        self._thread.join()

        return SystemResults(
            vram_usage=self._vram_usage,
            ram_usage=self._ram_usage,
            cpu_usage=self._cpu_usage,
        )
