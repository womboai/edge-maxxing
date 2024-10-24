import logging
import os
import sys
import time
from io import TextIOWrapper
from multiprocessing.connection import Client, Connection
from os.path import abspath
from pathlib import Path
from subprocess import Popen, run, TimeoutExpired, PIPE
from threading import Thread
from typing import Generic, TypeVar

from pydantic import BaseModel

from .setup_inference_sandbox import setup_sandbox, InvalidSubmissionError
from ..contest import ModelRepositoryInfo
from ..random_inputs import INFERENCE_SOCKET_TIMEOUT

logger = logging.getLogger(__name__)

RequestT = TypeVar("RequestT", bound=BaseModel)


class InferenceSandbox(Generic[RequestT]):
    _repository: ModelRepositoryInfo

    _client: Connection
    _process: Popen

    def __init__(self, repository_info: ModelRepositoryInfo, baseline: bool, sandbox_directory: Path, switch_user: bool, cache: bool):
        self._repository = repository_info
        self._baseline = baseline
        self._sandbox_directory = sandbox_directory
        self._switch_user = switch_user
        self._cache = cache

        try:
            self._file_size = setup_sandbox(
                self.sandbox_args(self._user),
                self._sandbox_directory,
                baseline,
                cache,
                repository_info.url,
                repository_info.revision,
            )
        except InvalidSubmissionError as e:
            if baseline:
                self.clear_sandbox()
                raise RuntimeError("Failed to setup baseline sandbox, cleared baseline sandbox directory") from e
            else:
                raise e

        logger.info(f"Repository {repository_info} had size {self._file_size}b")
        socket_path = abspath(self._sandbox_directory / "inferences.sock")
        self.remove_socket(socket_path)

        self._process = Popen(
            [
                *self.sandbox_args(self._user),
                abspath(self._sandbox_directory / ".venv" / "bin" / "start_inference")
            ],
            cwd=self._sandbox_directory,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
        )

        Thread(target=self._stream_logs, args=(self._process.stdout, sys.stdout), daemon=True).start()
        Thread(target=self._stream_logs, args=(self._process.stderr, sys.stderr), daemon=True).start()

        logger.info("Inference process starting")

        for _ in range(INFERENCE_SOCKET_TIMEOUT):
            if os.path.exists(socket_path):
                break

            time.sleep(1)

            self._check_exit()
        else:
            if baseline:
                self.clear_sandbox()
                raise RuntimeError(f"Baseline timed out after {INFERENCE_SOCKET_TIMEOUT} seconds. Cleared baseline sandbox directory")
            else:
                raise InvalidSubmissionError(f"Timed out after {INFERENCE_SOCKET_TIMEOUT} seconds")

        logger.info("Connecting to socket")
        try:
            self._client = Client(socket_path)
        except ConnectionRefusedError as e:
            if baseline:
                self.clear_sandbox()
                raise RuntimeError("Failed to connect to socket, cleared baseline sandbox directory") from e
            else:
                raise InvalidSubmissionError("Failed to connect to socket") from e

    @property
    def _user(self) -> str:
        return "baseline-sandbox" if self._baseline else "sandbox"

    def _check_exit(self):
        if self._process.returncode and not self._process.poll():
            if self._baseline:
                self.clear_sandbox()
                raise RuntimeError(f"Baseline inference crashed with exit code {self._process.returncode}. Cleared baseline sandbox directory")
            else:
                raise InvalidSubmissionError(f"Inference crashed with exit code {self._process.returncode}")

    def clear_sandbox(self):
        run(
            [
                *self.sandbox_args(self._user),
                "find",
                str(self._sandbox_directory),
                "-mindepth",
                "1",
                "-delete",
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )

    def remove_socket(self, socket_path: str):
        run(
            [
                *self.sandbox_args(self._user),
                "rm",
                "-f",
                socket_path,
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )

    def __enter__(self):
        self._process.__enter__()
        self._client.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.__exit__(exc_type, exc_val, exc_tb)

        self._process.terminate()

        try:
            self._process.wait(timeout=30)

            self._check_exit()
        except TimeoutExpired:
            self._process.kill()
            logger.warning(f"Forcefully killed inference process")

        self._process.__exit__(exc_type, exc_val, exc_tb)

        if not self._cache:
            self.clear_sandbox()

    def __call__(self, request: RequestT):
        self._check_exit()

        data = request.model_dump_json().encode("utf-8")

        self._client.send_bytes(data)

        return self._client.recv_bytes()

    @property
    def model_size(self):
        return self._file_size

    def sandbox_args(self, user: str) -> list[str]:
        return [
            "/bin/sudo",
            "-u",
            user,
        ] if self._switch_user else []

    @staticmethod
    def _stream_logs(stream: TextIOWrapper, output_stream: TextIOWrapper):
        for line in iter(stream.readline, ""):
            print(f"[INFERENCE] {line}", end="", file=output_stream)
