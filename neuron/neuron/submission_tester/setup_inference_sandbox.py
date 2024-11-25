import shutil
from os.path import abspath
from pathlib import Path
from subprocess import run, CalledProcessError

import toml
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi
from opentelemetry import trace

from ..checkpoint import SPEC_VERSION

DEPENDENCY_BLACKLIST = abspath(Path(__file__).parent / "dependency_blacklist.txt")

CLEAR_CACHE_SCRIPT = abspath(Path(__file__).parent / "clear_cache.sh")
CLONE_SCRIPT = abspath(Path(__file__).parent / "clone.sh")
BLACKLIST_SCRIPT = abspath(Path(__file__).parent / "blacklist.sh")
SYNC_UV = abspath(Path(__file__).parent / "sync_uv.sh")
DOWNLOAD_HUGGINGFACE_MODELS = abspath(Path(__file__).parent / "download_huggingface_models.sh")
NETWORK_JAIL = abspath(Path(__file__).parent / "libnetwork_jail.so")

STORAGE_THRESHOLD_GB = 50
MAX_HF_MODEL_SIZE_GB = 100
MAX_REPO_SIZE_MB = 16

with open(DEPENDENCY_BLACKLIST, 'r') as blacklist_file:
    BLACKLISTED_DEPENDENCIES = " ".join(blacklist_file.read().splitlines())

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
hf_api = HfApi()


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
            logger.info(process.stdout)
            logger.info(process.stderr)


def setup_sandbox(sandbox_args: list[str], sandbox_directory: Path, baseline: bool, url: str, revision: str) -> int:
    with tracer.start_as_current_span("setup_sandbox") as span:
        span.set_attributes({
            "repo.url": url,
            "repo.revision": revision
        })

        with tracer.start_as_current_span("check_disk_space") as disk_span:
            free_space = shutil.disk_usage("/").free
            disk_span.set_attribute("disk.free_space_gb", free_space / 1024 ** 3)
            if free_space < STORAGE_THRESHOLD_GB * 1024 ** 3:
                disk_span.set_attribute("cache.cleared", True)
                with tracer.start_span("clear_cache") as cache_span:
                    _run(
                        CLEAR_CACHE_SCRIPT,
                        sandbox_args,
                        sandbox_directory,
                        [],
                        "Failed to clear caches"
                    )
                    new_free_space = shutil.disk_usage("/").free
                    cache_span.set_attribute("cache.freed_space_gb", (new_free_space - free_space) / 1024 ** 3)

        with tracer.start_span("clone_repository") as clone_span:
            clone_span.set_attributes({
                "repo.url": url,
                "repo.revision": revision
            })

            _run(
                CLONE_SCRIPT,
                sandbox_args,
                sandbox_directory,
                [url, revision],
                "Failed to clone repository"
            )

            repo_size = sum(file.stat().st_size for file in sandbox_directory.rglob("*") if ".git" not in file.parts and ".venv" not in file.parts)
            clone_span.set_attribute("repo.size_mb", repo_size / 1024 ** 2)
            if repo_size > MAX_REPO_SIZE_MB * 1024 ** 2:
                raise InvalidSubmissionError(f"Size of repository exceeds {MAX_REPO_SIZE_MB} MB")

            try:
                with open(sandbox_directory / "pyproject.toml", 'r') as file:
                    pyproject = toml.load(file)
                    version = int(pyproject["project"]["version"])
                    models = pyproject["tool"]["edge-maxxing"]["models"]
            except Exception as e:
                raise InvalidSubmissionError("Failed to read submission info") from e

            clone_span.set_attribute("submission.version", version)
            if version != SPEC_VERSION:
                raise InvalidSubmissionError(f"Submission is at version {version} while expected version is {SPEC_VERSION}")

        if not baseline:
            with tracer.start_span("check_blacklist"):
                _run(
                    BLACKLIST_SCRIPT,
                    sandbox_args,
                    sandbox_directory,
                    [BLACKLISTED_DEPENDENCIES],
                    "Detected a blacklisted dependency"
                )

        with tracer.start_span("sync_uv"):
            _run(
                SYNC_UV,
                sandbox_args,
                sandbox_directory,
                [],
                "Failed to sync uv"
            )

        with tracer.start_span("download_huggingface_models") as hf_span:
            try:
                total_model_size = 0
                for model in models:
                    model_info = hf_api.model_info(repo_id=model, files_metadata=True)
                    for sibling in model_info.siblings:
                        total_model_size += sibling.size
            except Exception as e:
                raise InvalidSubmissionError("Failed to get model info") from e

            hf_span.set_attribute("models.total_size_gb", total_model_size / 1024 ** 3)
            if total_model_size > MAX_HF_MODEL_SIZE_GB * 1024 ** 3:
                raise InvalidSubmissionError(f"Size of all Hugging Face models exceeds {MAX_HF_MODEL_SIZE_GB} GB")

            _run(
                DOWNLOAD_HUGGINGFACE_MODELS,
                sandbox_args,
                sandbox_directory,
                [" ".join(models)],
                "Failed to download Hugging Face models"
            )

        return repo_size + total_model_size
