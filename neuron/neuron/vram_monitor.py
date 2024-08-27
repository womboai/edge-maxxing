import threading
import time

import pynvml
import torch

from contest import Contest

POLL_RATE_SECONDS = 0.1


class VRamMonitor:
    _thread: threading.Thread
    _device: torch.device
    _contest: Contest
    _vram_usage: set[int] = set()
    _stop_flag: threading.Event
    _lock: threading.Lock

    def __init__(self, contest: Contest, device: torch.device):
        self._contest = contest
        self._device = device
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self.monitor)
        self._thread.start()

    def monitor(self):
        pynvml.nvmlInit()
        while not self._stop_flag.is_set():
            vram = self._contest.get_vram_used(self._device)
            with self._lock:
                self._vram_usage.add(vram)
            time.sleep(POLL_RATE_SECONDS)
        pynvml.nvmlShutdown()

    def complete(self) -> float:
        self._stop_flag.set()
        self._thread.join()
        with self._lock:
            return sum(self._vram_usage) / len(self._vram_usage)
