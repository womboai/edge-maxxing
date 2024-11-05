import sys
from logging import getLogger
from os.path import abspath
from pathlib import Path
from subprocess import run, CalledProcessError
from time import perf_counter
from ..checkpoint import SPEC_VERSION
import shutil
import toml
import os
from huggingface_hub import HfApi

DEPENDENCY_BLACKLIST = abspath(Path(__file__).parent / "dependency_blacklist.txt")

CLEAR_CACHE_SCRIPT = abspath(Path(__file__).parent / "clear_cache.sh")
CLONE_SCRIPT = abspath(Path(__file__).parent / "clone.sh")
BLACKLIST_SCRIPT = abspath(Path(__file__).parent / "blacklist.sh")
DOWNLOAD_HUGGINGFACE_MODELS = abspath(Path(__file__).parent / "download_huggingface_models.sh")
NETWORK_JAIL = abspath(Path(__file__).parent / "libnetwork_jail.so")

STORAGE_THRESHOLD_GB = 50
MAX_HF_MODEL_SIZE_GB = 50

with open(DEPENDENCY_BLACKLIST, 'r') as blacklist_file:
    BLACKLISTED_DEPENDENCIES = " ".join(blacklist_file.read().splitlines())

logger = getLogger(__name__)
hf_api = HfApi()
debug = int(os.getenv("VALIDATOR_DEBUG") or 0) > 0


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
            capture_output=not debug,
            stdout=sys.stdout if debug else None,
            stderr=sys.stderr if debug else None,
            encoding='utf-8',
            cwd=sandbox_directory.absolute(),
        )
        process.check_returncode()
    except CalledProcessError as e:
        raise InvalidSubmissionError(error_message) from e
    finally:
        if process and not debug:
            if process.stdout.strip():
                print(process.stdout)
            if process.stderr.strip():
                print(process.stderr, file=sys.stderr)


def setup_sandbox(sandbox_args: list[str], sandbox_directory: Path, baseline: bool, url: str, revision: str) -> int:
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
        [url, revision],
        "Failed to clone repository"
    )
    logger.info(f"Cloned repository '{url}' in {perf_counter() - start:.2f} seconds")

    try:
        with open(sandbox_directory / "pyproject.toml", 'r') as file:
            pyproject = toml.load(file)
            version = int(pyproject["project"]["version"])
            models = pyproject["tool"]["edge-maxxing"]["models"]
    except Exception as e:
        raise InvalidSubmissionError("Failed to read submission info") from e

    if version != SPEC_VERSION:
        raise InvalidSubmissionError(f"Submission is at version {version} while expected version is {SPEC_VERSION}")

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
    logger.info(f"Downloading Hugging Face models...")
    try:
        total_model_size = 0
        for model in models:
            model_info = hf_api.model_info(repo_id=model, files_metadata=True)
            for sibling in model_info.siblings:
                total_model_size += sibling.size
    except Exception as e:
        raise InvalidSubmissionError("Failed to get model info") from e

    if total_model_size > MAX_HF_MODEL_SIZE_GB * 1024 ** 3:
        raise InvalidSubmissionError(f"Size of all Hugging Face models exceeds {MAX_HF_MODEL_SIZE_GB} GB")

    _run(
        DOWNLOAD_HUGGINGFACE_MODELS,
        sandbox_args,
        sandbox_directory,
        [" ".join(models)],
        "Failed to download Hugging Face models"
    )
    logger.info(f"Downloaded Hugging Face model in {perf_counter() - start:.2f} seconds")

    return sum(
        file.stat().st_size for file in sandbox_directory.rglob("*")
        if ".git" not in file.parts and ".venv" not in file.parts
    ) + total_model_size
