import os
import signal
from threading import Event, Thread
from time import localtime, sleep

import git
from fiber.logging_utils import get_logger

UPDATE_RATE_MINUTES = 10

logger = get_logger(__name__)


class AutoUpdater:
    _thread: Thread
    _stop_flag: Event

    def __init__(self):
        self._stop_flag = Event()
        self._thread = Thread(target=self._monitor)
        self._check_for_updates()
        self._thread.start()

    def _monitor(self):
        while not self._stop_flag.is_set():
            try:
                current_time = localtime()
                if current_time.tm_min % UPDATE_RATE_MINUTES == 0:
                    self._check_for_updates()
                    self._stop_flag.wait(60)
                else:
                    sleep_minutes = UPDATE_RATE_MINUTES - current_time.tm_min % UPDATE_RATE_MINUTES
                    self._stop_flag.wait(sleep_minutes * 60 - current_time.tm_sec)
            except Exception as e:
                logger.error("Error occurred while checking for updates, attempting to fix the issue by restarting", exc_info=e)
                self._stop_flag.wait(1)
                self._restart()

    def _check_for_updates(self):
        logger.info("Checking for updates...")
        repo = git.Repo(search_parent_directories=True)
        current_version = repo.head.commit.hexsha

        try:
            if repo.is_dirty():
                logger.info("Stashing local changes...")
                repo.git.stash(include_untracked=True)
        except Exception as e:
            logger.error("Unable to stash local changes", exc_info=e)

        repo.git.pull()

        new_version = repo.head.commit.hexsha

        if current_version != new_version:
            logger.info(f"New version detected: '{new_version}'. Restarting...")
            self._restart()
        else:
            logger.info("Already up to date.")

    def shutdown(self):
        self._stop_flag.set()

    def _restart(self):
        self.shutdown()
        os.kill(os.getpid(), signal.SIGTERM)

        logger.info("Waiting for process to terminate...")
        for _ in range(60):
            sleep(1)
            try:
                os.kill(os.getpid(), 0)
            except ProcessLookupError:
                break

        os.kill(os.getpid(), signal.SIGKILL)
