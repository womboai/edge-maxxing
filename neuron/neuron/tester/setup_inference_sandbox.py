from fiber.logging_utils import get_logger
import sys
from os.path import abspath
from pathlib import Path
from subprocess import run, CalledProcessError
from time import perf_counter

DEPENDENCY_BLACKLIST = abspath(Path(__file__).parent / "dependency_blacklist.txt")

CLONE_SCRIPT = abspath(Path(__file__).parent / "clone.sh")
BLACKLIST_SCRIPT = abspath(Path(__file__).parent / "blacklist.sh")
LFS_PULL_SCRIPT = abspath(Path(__file__).parent / "lfs_pull.sh")
PIP_INSTALL_SCRIPT = abspath(Path(__file__).parent / "pip_install.sh")

with open(DEPENDENCY_BLACKLIST, 'r') as blacklist_file:
    BLACKLISTED_DEPENDENCIES = " ".join(blacklist_file.read().splitlines())

logger = get_logger(__name__)


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
            print(process.stdout)
            print(process.stderr, file=sys.stderr)


def setup_sandbox(sandbox_args: list[str], sandbox_directory: Path, baseline: bool, url: str, revision: str) -> int:
    start = perf_counter()
    logger.info(f"Cloning repository '{url}' with revision '{revision}'...")
    _run(
        CLONE_SCRIPT,
        sandbox_args,
        sandbox_directory,
        [url, revision, str(baseline).lower()],
        f"Failed to clone repository '{url}'"
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
            f"Found blacklisted dependency in repository '{url}'"
        )
        logger.info(f"Checked for blacklisted dependencies in {perf_counter() - start:.2f} seconds")

    start = perf_counter()
    logger.info(f"Pulling LFS files...")
    _run(
        LFS_PULL_SCRIPT,
        sandbox_args,
        sandbox_directory,
        [],
        f"Failed to pull LFS files from repository '{url}'"
    )
    logger.info(f"Pulled LFS files in {perf_counter() - start:.2f} seconds")

    file_size = sum(file.stat().st_size for file in sandbox_directory.rglob("*"))

    start = perf_counter()
    logger.info(f"Installing dependencies...")
    _run(
        PIP_INSTALL_SCRIPT,
        sandbox_args,
        sandbox_directory,
        [],
        f"Failed to install dependencies from repository '{url}'"
    )
    logger.info(f"Installed dependencies in {perf_counter() - start:.2f} seconds")

    return file_size