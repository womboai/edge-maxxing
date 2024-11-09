import os
import time
from os.path import abspath
from pathlib import Path
from threading import Event, Thread
from subprocess import run

AUTO_UPDATE_SCRIPT = abspath(Path(__file__).parent / "auto-update.sh")
UPDATE_RATE_MINUTES = 15

class AutoUpdater:
    _thread: Thread
    _stop_flag: Event

    def __init__(self):
        self._stop_flag = Event()
        self._thread = Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def _monitor(self):
        while not self._stop_flag.is_set():
            current_time = time.localtime()
            if current_time.tm_min % UPDATE_RATE_MINUTES == 0:
                self._check_for_updates()
                time.sleep(60)
            else:
                sleep_minutes = UPDATE_RATE_MINUTES - current_time.tm_min % UPDATE_RATE_MINUTES
                time.sleep(sleep_minutes * 60 - current_time.tm_sec)

    def _check_for_updates(self):
        process = run(
            [AUTO_UPDATE_SCRIPT],
            capture_output=True,
            encoding='utf-8',
        )

        if process.stdout.strip():
            print(process.stdout)

        if process.returncode == 75:
            self._restart()

    def _restart(self):
        self._stop_flag.set()
        os._exit(75)
