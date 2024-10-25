import json
import sys
from logging import getLogger
from os.path import abspath
from pathlib import Path
from subprocess import run, CalledProcessError
from time import perf_counter
import shutil

DEPENDENCY_BLACKLIST = abspath(Path(__file__).parent / "dependency_blacklist.txt")

CLEAR_CACHE_SCRIPT = abspath(Path(__file__).parent / "clear_cache.sh")
CLONE_SCRIPT = abspath(Path(__file__).parent / "clone.sh")
BLACKLIST_SCRIPT = abspath(Path(__file__).parent / "blacklist.sh")
LFS_PULL_SCRIPT = abspath(Path(__file__).parent / "lfs_pull.sh")
PIP_INSTALL_SCRIPT = abspath(Path(__file__).parent / "pip_install.sh")

STORAGE_THRESHOLD_GB = 50

with open(DEPENDENCY_BLACKLIST, 'r') as blacklist_file:
    BLACKLISTED_DEPENDENCIES = " ".join(blacklist_file.read().splitlines())

logger = getLogger(__name__)


class InvalidSubmissionError(Exception):
    ...


def _run(script: str, sandbox_args: list[str], sandbox_directory: Path, args: list[str], error_message: str):
    process = None
    try:
        process = run(
            [
                *sandbox_args,
                script,
                *args,
            ],
            capture_output=True,
            encoding='utf-8',
            cwd=sandbox_directory.absolute(),
        )
        process.check_returncode()
    except CalledProcessError as e:
        raise InvalidSubmissionError(error_message) from e
    finally:
        if process:
            if process.stdout.strip():
                print(process.stdout)
            if process.stderr.strip():
                print(process.stderr, file=sys.stderr)

def is_cached(sandbox_directory: Path, url: str, revision: str) -> bool:
    cache_file = sandbox_directory / "cache_info.json"
    if not cache_file.exists():
        return False

    with open(cache_file, 'r') as file:
        cache_info = json.load(file)
        return cache_info["repository"] == url and cache_info["revision"] == revision


def setup_sandbox(sandbox_args: list[str], sandbox_directory: Path, baseline: bool, cache: bool, url: str, revision: str) -> int:
    free_space = shutil.disk_usage("/").free
    if free_space < STORAGE_THRESHOLD_GB * 1024 ** 3:
        logger.info(f"Running low on disk space: {free_space / 1024 ** 3:.2f} GB remaining. Clearing caches...")
        _run(
            CLEAR_CACHE_SCRIPT,
            sandbox_args,
            sandbox_directory,
            [],
            "Failed to clear caches"
        )
        new_free_space = shutil.disk_usage("/").free
        logger.info(f"Cleared {(new_free_space - free_space) / 1024 ** 3:.2f} GB of caches")

    start = perf_counter()
    logger.info(f"Cloning repository '{url}' with revision '{revision}'...")
    _run(
        CLONE_SCRIPT,
        sandbox_args,
        sandbox_directory,
        [url, revision, str(cache).lower(), str(is_cached(sandbox_directory, url, revision)).lower()],
        "Failed to clone repository"
    )
    logger.info(f"Cloned repository '{url}' in {perf_counter() - start:.2f} seconds")

    if not baseline:
        start = perf_counter()
        logger.info(f"Checking for blacklisted dependencies...")
        _run(
            BLACKLIST_SCRIPT,
            sandbox_args,
            sandbox_directory,
            [BLACKLISTED_DEPENDENCIES],
            "Detected a blacklisted dependency"
        )
        logger.info(f"Found no blacklisted dependencies after {perf_counter() - start:.2f} seconds")

    start = perf_counter()
    logger.info(f"Pulling LFS files...")
    _run(
        LFS_PULL_SCRIPT,
        sandbox_args,
        sandbox_directory,
        [],
        "Failed to pull LFS files"
    )
    logger.info(f"Pulled LFS files in {perf_counter() - start:.2f} seconds")

    file_size = sum(
        file.stat().st_size for file in sandbox_directory.rglob("*")
        if ".git" not in file.parts and ".venv" not in file.parts
    )

    start = perf_counter()
    logger.info(f"Installing dependencies...")
    _run(
        PIP_INSTALL_SCRIPT,
        sandbox_args,
        sandbox_directory,
        [],
        "Failed to install dependencies"
    )
    logger.info(f"Installed dependencies in {perf_counter() - start:.2f} seconds")

    return file_size
