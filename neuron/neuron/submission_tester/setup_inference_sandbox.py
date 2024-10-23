import json
import sys
from logging import getLogger
from os.path import abspath
from pathlib import Path
from subprocess import run, CalledProcessError
from time import perf_counter

DEPENDENCY_BLACKLIST = abspath(Path(__file__).parent / "dependency_blacklist.txt")

CLONE_SCRIPT = abspath(Path(__file__).parent / "clone.sh")
BLACKLIST_SCRIPT = abspath(Path(__file__).parent / "blacklist.sh")
LFS_PULL_SCRIPT = abspath(Path(__file__).parent / "lfs_pull.sh")
POETRY_INSTALL_SCRIPT = abspath(Path(__file__).parent / "poetry_install.sh")
CACHE_SCRIPT = abspath(Path(__file__).parent / "cache.sh")

with open(DEPENDENCY_BLACKLIST, 'r') as blacklist_file:
    BLACKLISTED_DEPENDENCIES = " ".join(blacklist_file.read().splitlines())

logger = getLogger(__name__)


class InvalidSubmissionError(Exception):
    ...


def _run(script: str, sandbox_args: list[str], sandbox_directory: Path, args: list[str], error_message: str):
    try:
        run(
            [
                *sandbox_args,
                script,
                *args,
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=sandbox_directory.absolute(),
            check=True,
        )
    except CalledProcessError as e:
        raise InvalidSubmissionError(error_message) from e


def is_cached(sandbox_directory: Path, url: str, revision: str) -> bool:
    cache_file = sandbox_directory / "cache_info.json"
    if not cache_file.exists():
        return False

    with open(cache_file, 'r') as file:
        cache_info = json.load(file)
        return cache_info["repository"] == url and cache_info["revision"] == revision


def get_submission_size(sandbox_directory: Path) -> int:
    return sum(
        file.stat().st_size for file in sandbox_directory.rglob("*")
        if ".git" not in file.parts and ".venv" not in file.parts
    )


def setup_sandbox(sandbox_args: list[str], sandbox_directory: Path, baseline: bool, url: str, revision: str) -> int:
    if is_cached(sandbox_directory, url, revision):
        logger.info(f"Using cached repository")
        return get_submission_size(sandbox_directory)

    start = perf_counter()
    logger.info(f"Cloning repository '{url}' with revision '{revision}'...")
    _run(
        CLONE_SCRIPT,
        sandbox_args,
        sandbox_directory,
        [url, revision],
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

    start = perf_counter()
    logger.info(f"Installing dependencies...")
    _run(
        POETRY_INSTALL_SCRIPT,
        sandbox_args,
        sandbox_directory,
        [],
        "Failed to install dependencies"
    )
    logger.info(f"Installed dependencies in {perf_counter() - start:.2f} seconds")

    _run(
        CACHE_SCRIPT,
        sandbox_args,
        sandbox_directory,
        [url, revision],
        "Failed to create cache file"
    )

    return get_submission_size(sandbox_directory)
