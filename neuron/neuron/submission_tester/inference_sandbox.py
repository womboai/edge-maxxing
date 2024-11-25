import os
from io import TextIOWrapper
from multiprocessing.connection import Client, Connection
from os.path import abspath
from pathlib import Path
from subprocess import Popen, TimeoutExpired, PIPE
from threading import Thread
from time import perf_counter, sleep
from typing import Generic, TypeVar

from fiber.logging_utils import get_logger
from opentelemetry import trace
from pydantic import BaseModel

from .setup_inference_sandbox import setup_sandbox, InvalidSubmissionError, NETWORK_JAIL
from ..contest import ModelRepositoryInfo

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

RequestT = TypeVar("RequestT", bound=BaseModel)

SANDBOX = "sandbox"
START_INFERENCE_SCRIPT = abspath(Path(__file__).parent / "start_inference.sh")


class InferenceSandbox(Generic[RequestT]):
    _repository: ModelRepositoryInfo

    _client: Connection
    _process: Popen
    load_time: float

    @tracer.start_as_current_span("init_inference_sandbox")
    def __init__(
        self,
        repository_info: ModelRepositoryInfo,
        baseline: bool,
        sandbox_directory: Path,
        switch_user: bool,
        load_timeout: int,
    ):
        self._repository = repository_info
        self._baseline = baseline
        self._sandbox_directory = sandbox_directory
        self._switch_user = switch_user

        try:
            with tracer.start_span("setup_sandbox"):
                self._file_size = setup_sandbox(
                    sandbox_args=self.sandbox_args(SANDBOX),
                    sandbox_directory=self._sandbox_directory,
                    baseline=baseline,
                    url=repository_info.url,
                    revision=repository_info.revision,
                )
        except InvalidSubmissionError as e:
            self.fail(str(e))

        with tracer.start_span("start_inference_process") as inference_span:
            self._process = Popen(
                [
                    *self.sandbox_args(SANDBOX),
                    START_INFERENCE_SCRIPT,
                    NETWORK_JAIL
                ],
                cwd=self._sandbox_directory,
                stdout=PIPE,
                stderr=PIPE,
                text=True,
            )

            Thread(target=self._stream_logs, args=(self._process.stdout,), daemon=True).start()
            Thread(target=self._stream_logs, args=(self._process.stderr,), daemon=True).start()

            logger.info("Inference process starting")

            socket_path = abspath(self._sandbox_directory / "inferences.sock")
            start = perf_counter()
            for _ in range(load_timeout):
                if os.path.exists(socket_path): break
                sleep(1)
                self._check_exit()
            else:
                self.fail(f"Timed out after {load_timeout} seconds")

            logger.info("Connecting to socket")
            try:
                self._client = Client(socket_path)
            except ConnectionRefusedError:
                self.fail("Failed to connect to socket")
            self.load_time = perf_counter() - start
            inference_span.set_attribute("load_time_seconds", self.load_time)

    def _check_exit(self):
        if self._process.poll():
            self.fail(f"Inference crashed with exit code {self._process.returncode}")

    def __enter__(self):
        self._process.__enter__()
        self._client.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.__exit__(exc_type, exc_val, exc_tb)

        self._process.terminate()

        try:
            self._process.wait(timeout=30)
        except TimeoutExpired:
            self._process.kill()
            logger.warning(f"Forcefully killed inference process")

        self._process.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, request: RequestT):
        self._check_exit()

        data = request.model_dump_json().encode("utf-8")

        self._client.send_bytes(data)

        return self._client.recv_bytes()

    @property
    def model_size(self):
        return self._file_size

    def fail(self, reason: str):
        raise InvalidSubmissionError(reason) if not self._baseline else RuntimeError(reason)

    def sandbox_args(self, user: str) -> list[str]:
        return [
            "/bin/sudo",
            "-u",
            user,
        ] if self._switch_user else []

    @staticmethod
    def _stream_logs(stream: TextIOWrapper):
        for line in iter(stream.readline, ""):
            logger.info(line.rstrip())
