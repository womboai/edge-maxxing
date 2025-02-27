import shutil
from concurrent.futures import CancelledError
from multiprocessing.connection import Client, wait
from os.path import abspath
from pathlib import Path
from subprocess import run, Popen, PIPE, TimeoutExpired
from threading import Event
from time import perf_counter
from typing import Annotated

import toml
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi
from opentelemetry import trace
from pydantic import BaseModel, Field

from base.checkpoint import SPEC_VERSION
from base.contest import RepositoryInfo, Contest, Metrics
from base.inputs_api import get_blacklist
from pipelines import TextToImageRequest
from .system_monitor import SystemMonitor

CLEAR_CACHE = abspath(Path(__file__).parent / "clear_cache.sh")
CLONE = abspath(Path(__file__).parent / "clone.sh")
BLACKLIST = abspath(Path(__file__).parent / "blacklist.sh")
SYNC_UV = abspath(Path(__file__).parent / "sync_uv.sh")
DOWNLOAD_HUGGINGFACE_MODELS = abspath(Path(__file__).parent / "download_huggingface_models.py")

NETWORK_JAIL = abspath(Path(__file__).parent / "libnetwork_jail.so")
START_INFERENCE_SCRIPT = abspath(Path(__file__).parent / "start_inference.sh")

STORAGE_THRESHOLD_GB = 50
MAX_HF_MODEL_SIZE_GB = 100
MAX_REPO_SIZE_MB = 16

LOAD_TIMEOUT = 240
EXIT_TIMEOUT = 30

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
hf_api = HfApi()


class InvalidSubmissionError(Exception):
    ...


class BenchmarkOutput(BaseModel):
    metrics: Metrics
    outputs: list[bytes]


class ModelSpecification(BaseModel):
    repository: str
    revision: str
    include: Annotated[list[str], Field(default_factory=lambda: [])]
    exclude: Annotated[list[str], Field(default_factory=lambda: [])]


class InferenceSandbox:
    _sandbox_args: list[str]
    _sandbox_directory: Path
    _contest: Contest
    _repository_info: RepositoryInfo
    _inputs: list[TextToImageRequest]
    _stop_flag: Event
    _socket_path: Path

    def __init__(
        self,
        sandbox_args: list[str],
        sandbox_directory: Path,
        contest: Contest,
        repository_info: RepositoryInfo,
        inputs: list[TextToImageRequest],
        stop_flag: Event,
    ):
        self._sandbox_args = sandbox_args
        self._sandbox_directory = sandbox_directory
        self._contest = contest
        self._repository_info = repository_info
        self._inputs = inputs
        self._stop_flag = stop_flag
        self._socket_path = self._sandbox_directory / "inferences.sock"

    def _run(self, script: str, args: list[str]):
        if self._stop_flag.is_set():
            raise CancelledError()

        process = run(
            [
                *self._sandbox_args,
                script,
                *args,
            ],
            capture_output=True,
            encoding='utf-8',
            cwd=self._sandbox_directory.absolute(),
        )
        if process.stdout.strip():
            logger.info(process.stdout)
        if process.stderr.strip():
            logger.info(process.stderr)
        if process.returncode:
            raise InvalidSubmissionError(f"Failed to run {script}")

    @tracer.start_as_current_span("check_space")
    def _check_space(self):
        free_space = shutil.disk_usage("/").free
        if free_space < STORAGE_THRESHOLD_GB * 1024 ** 3:
            logger.info(f"{free_space / 1024 ** 3:.2f} GB of free space left, clearing caches")
            self._run(CLEAR_CACHE, [])
            new_free_space = shutil.disk_usage("/").free
            logger.info(f"Cleared {(new_free_space - free_space) / 1024 ** 3:.2f} GB of caches. {new_free_space / 1024 ** 3:.2f} GB of free space left")

    @tracer.start_as_current_span("clone_repository")
    def _clone(self) -> int:
        self._run(CLONE, [self._repository_info.url, self._repository_info.revision])

        repo_size = sum(
            file.stat().st_size for file in self._sandbox_directory.rglob("*")
            if ".git" not in file.parts and ".venv" not in file.parts
        )
        if repo_size > MAX_REPO_SIZE_MB * 1024 ** 2:
            raise InvalidSubmissionError(f"Size of repository exceeds {MAX_REPO_SIZE_MB} MB")

        return repo_size

    @tracer.start_as_current_span("check_blacklist")
    def _check_blacklist(self):
        self._run(BLACKLIST, list(get_blacklist().dependencies))

    @tracer.start_as_current_span("sync_uv")
    def _sync_uv(self):
        self._run(SYNC_UV, [])

    @tracer.start_as_current_span("download_huggingface_models")
    def _download_huggingface_models(self, models: list[ModelSpecification]) -> int:
        try:
            total_model_size = 0
            for model in models:
                model_info = hf_api.model_info(model.repository, revision=model.revision, files_metadata=True)
                for sibling in model_info.siblings:
                    total_model_size += sibling.size
        except Exception as e:
            raise InvalidSubmissionError("Failed to get model info") from e

        if total_model_size > MAX_HF_MODEL_SIZE_GB * 1024 ** 3:
            raise InvalidSubmissionError(f"Size of all Hugging Face models exceeds {MAX_HF_MODEL_SIZE_GB} GB")

        python_executable = str((self._sandbox_directory / ".venv" / "bin" / "python").absolute())

        self._run(python_executable, [DOWNLOAD_HUGGINGFACE_MODELS] + [model.model_dump_json() for model in models])

        return total_model_size

    @tracer.start_as_current_span("setup")
    def _setup_sandbox(self) -> int:
        self._check_space()
        repository_size = self._clone()

        try:
            with open(self._sandbox_directory / "pyproject.toml", 'r') as file:
                pyproject = toml.load(file)
                version = int(pyproject["project"]["version"])
                models = list(map(ModelSpecification.model_validate, pyproject["tool"]["edge-maxxing"]["models"]))
        except Exception as e:
            raise InvalidSubmissionError("Failed to read submission info") from e

        if version != SPEC_VERSION:
            raise InvalidSubmissionError(f"Submission is at version {version} while expected version is {SPEC_VERSION}")

        self._check_blacklist()
        self._sync_uv()
        huggingface_models_size = self._download_huggingface_models(models)

        return repository_size + huggingface_models_size

    @tracer.start_as_current_span("wait_for_socket")
    def wait_for_socket(self, process: Popen) -> float:
        start = perf_counter()
        for _ in range(LOAD_TIMEOUT):
            if self._socket_path.exists():
                break
            self._stop_flag.wait(1)
            check_process(process)
            if self._stop_flag.is_set():
                raise CancelledError()
        else:
            raise InvalidSubmissionError(f"Timed out after {LOAD_TIMEOUT} seconds")
        return perf_counter() - start

    @tracer.start_as_current_span("start_inference")
    def benchmark(self, timeout: float) -> BenchmarkOutput:
        size = self._setup_sandbox()
        start_vram = self._contest.device.get_vram_used()

        metrics: list[Metrics] = []
        outputs: list[bytes] = []

        logger.info("Loading pipeline")
        with Popen(
            [
                *self._sandbox_args,
                START_INFERENCE_SCRIPT,
                NETWORK_JAIL
            ],
            cwd=self._sandbox_directory,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            bufsize=1,
        ) as process:
            try:
                logger.info("Waiting for socket")
                load_time = self.wait_for_socket(process)
                logger.info("Starting inference")
                with Client(abspath(self._socket_path)) as client:
                    logger.info(f"Benchmarking {len(self._inputs)} samples")
                    for i, request in enumerate(self._inputs):
                        logger.info(f"Sample {i + 1}/{len(self._inputs)}")
                        start_joules = self._contest.device.get_joules()
                        system_monitor = SystemMonitor(self._contest, process.pid)

                        data = request.model_dump_json().encode("utf-8")
                        logger.debug(data)
                        client.send_bytes(data)

                        start = perf_counter()

                        if wait([client], timeout=timeout):
                            output = client.recv_bytes()
                        else:
                            raise InvalidSubmissionError(f"Inference timed out after {timeout} seconds")

                        generation_time = perf_counter() - start
                        joules_used = self._contest.device.get_joules() - start_joules
                        watts_used = joules_used / generation_time
                        results = system_monitor.complete()

                        metrics.append(Metrics(
                            generation_time=generation_time,
                            size=size,
                            vram_used=results.vram_usage,
                            watts_used=watts_used,
                            load_time=load_time,
                            ram_used=results.ram_usage,
                        ))
                        outputs.append(output)
                        check_process(process)
            finally:
                logger.info("Exiting inference sandbox")
                try:
                    log_process(process)
                    process.terminate()
                    process.wait(timeout=EXIT_TIMEOUT)
                except TimeoutExpired:
                    logger.info("Inference sandbox did not exit gracefully, killing...")
                    process.kill()

        average_generation_time = sum(metric.generation_time for metric in metrics) / len(metrics)
        vram_used = max(metric.vram_used for metric in metrics) - start_vram
        ram_used = max(metric.ram_used for metric in metrics)
        watts_used = max(metric.watts_used for metric in metrics)
        return BenchmarkOutput(
            metrics=Metrics(
                generation_time=average_generation_time,
                size=size,
                vram_used=vram_used,
                watts_used=watts_used,
                load_time=load_time,
                ram_used=ram_used,
            ),
            outputs=outputs,
        )


def check_process(process: Popen):
    if process.poll():
        raise InvalidSubmissionError(f"Inference crashed with exit code {process.returncode}")


def log_process(process: Popen):
    try:
        stdout, stderr = process.communicate(timeout=EXIT_TIMEOUT)
        if stdout.strip():
            logger.info(f"STDOUT: {stdout}")
        if stderr.strip():
            logger.info(f"STDERR: {stderr}")
    except TimeoutExpired:
        logger.error("Timed out while reading logs")
