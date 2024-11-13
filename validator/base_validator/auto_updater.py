import os
import time
from os.path import abspath
from pathlib import Path
from threading import Event, Thread
from fiber.logging_utils import get_logger
import git

AUTO_UPDATE_SCRIPT = abspath(Path(__file__).parent / "auto-update.sh")
UPDATE_RATE_MINUTES = 15

logger = get_logger(__name__)

class AutoUpdater:
    _thread: Thread
    _stop_flag: Event

    def __init__(self):
        self._stop_flag = Event()
        self._thread = Thread(target=self._monitor, daemon=True)
        self._check_for_updates()
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
        logger.info("Checking for updates...")
        repo = git.Repo(search_parent_directories=True)
        current_version = repo.head.commit.hexsha

        with repo.git.custom_environment(GIT_AUTO_STASH='1'):
            repo.remotes.origin.pull("main")

        new_version = repo.head.commit.hexsha

        if current_version != new_version:
            logger.info(f"New version detected: '{new_version}'. Restarting...")
            self._restart()
        else:
            logger.info("Already up to date.")


    def _restart(self):
        self._stop_flag.set()
        os._exit(75)
