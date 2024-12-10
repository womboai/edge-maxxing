from dataclasses import dataclass
from threading import Thread, Event
from fiber.logging_utils import get_logger
import psutil

from base.contest import Contest

SAMPLE_RATE_MS = 100

logger = get_logger(__name__)


@dataclass
class SystemResults:
    vram_usage: int
    ram_usage: int


class SystemMonitor:
    _contest: Contest
    _process: psutil.Process
    _thread: Thread
    _stop_flag: Event

    _vram_usage: int = 0
    _ram_usage: int = 0

    def __init__(self, contest: Contest, pid: int):
        self._contest = contest
        self._process = psutil.Process(pid)
        self._stop_flag = Event()

        self._thread = Thread(target=self._monitor)
        self._thread.start()

    def _monitor(self):
        while not self._stop_flag.is_set():
            self._vram_usage = max(self._vram_usage, self._contest.device.get_vram_used())

            ram = self._process.memory_info().rss
            ram += sum([child.memory_info().rss for child in self._process.children(recursive=True)])
            self._ram_usage = max(self._ram_usage, ram)

            self._stop_flag.wait(SAMPLE_RATE_MS / 1000)

    def complete(self) -> SystemResults:
        self._stop_flag.set()
        self._thread.join()

        return SystemResults(
            vram_usage=self._vram_usage,
            ram_usage=self._ram_usage,
        )
