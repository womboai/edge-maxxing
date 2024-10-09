import logging
import os
import sys
import time
from io import TextIOWrapper
from multiprocessing.connection import Client
from os.path import abspath
from pathlib import Path
from subprocess import Popen, run, TimeoutExpired, CalledProcessError, PIPE
from typing import Generic
from threading import Thread

from neuron import (
    RequestT,
    INFERENCE_SOCKET_TIMEOUT,
    ModelRepositoryInfo,
    setup_sandbox,
    InvalidSubmissionError,
)

SANDBOX_DIRECTORY = Path("/sandbox")
BASELINE_SANDBOX_DIRECTORY = Path("/baseline-sandbox")

logger = logging.getLogger(__name__)


def sandbox_args(user: str):
    return [
        "/bin/sudo",
        "-u",
        user,
    ]


class InferenceSandbox(Generic[RequestT]):
    _repository: ModelRepositoryInfo

    _client: Client
    _process: Popen

    def __init__(self, repository_info: ModelRepositoryInfo, baseline: bool):
        self._repository = repository_info
        self._baseline = baseline

        try:
            self._file_size = setup_sandbox(sandbox_args(self._user), self._sandbox_directory, baseline, repository_info.url,repository_info.revision)
        except InvalidSubmissionError as e:
            if baseline:
                self.clear_sandbox()
                raise RuntimeError(f"Failed to setup baseline sandbox, cleared baseline sandbox directory") from e
            else:
                raise e

        logger.info(f"Repository {repository_info} had size {self._file_size}")

        self._process = Popen(
            [
                *sandbox_args(self._user),
                abspath(self._sandbox_directory / ".venv" / "bin" / "start_inference")
            ],
            cwd=self._sandbox_directory,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
        )

        Thread(target=self._stream_logs, args=(self._process.stdout, "STDOUT", sys.stdout), daemon=True).start()
        Thread(target=self._stream_logs, args=(self._process.stderr, "STDERR", sys.stderr), daemon=True).start()

        logger.info(f"Inference process starting")
        socket_path = abspath(self._sandbox_directory / "inferences.sock")

        for _ in range(INFERENCE_SOCKET_TIMEOUT):
            if os.path.exists(socket_path):
                break

            time.sleep(1)

            self._check_exit()
        else:
            raise InvalidSubmissionError(f"Socket file '{socket_path}' not found after {INFERENCE_SOCKET_TIMEOUT} seconds.")

        logger.info(f"Connecting to socket")
        try:
            self._client = Client(socket_path)
        except ConnectionRefusedError as e:
            if baseline:
                self.clear_sandbox()
                raise InvalidSubmissionError(f"Failed to connect to socket, cleared baseline sandbox directory") from e

    @property
    def _user(self) -> str:
        return "baseline-sandbox" if self._baseline else "sandbox"

    @property
    def _sandbox_directory(self) -> Path:
        return BASELINE_SANDBOX_DIRECTORY if self._baseline else SANDBOX_DIRECTORY

    def _check_exit(self):
        if self._process.returncode:
            raise InvalidSubmissionError(f"'{self._repository}'s inference crashed, got exit code {self._process.returncode}")

    def clear_sandbox(self):
        process = run(
            [
                *sandbox_args(self._user),
                "find",
                str(self._sandbox_directory),
                "-mindepth",
                "1",
                "-delete",
            ],
            capture_output=True,
            encoding='utf-8',
        )

        print(process.stdout)
        print(process.stderr, file=sys.stderr)
        process.check_returncode()

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

        if not self._baseline:
            self.clear_sandbox()

    def __call__(self, request: RequestT):
        self._check_exit()

        data = request.model_dump_json().encode("utf-8")

        self._client.send_bytes(data)

        return self._client.recv_bytes()

    @staticmethod
    def _stream_logs(stream: TextIOWrapper, prefix: str, output_stream: TextIOWrapper):
        for line in iter(stream.readline, ""):
            print(f"[INFERENCE - {prefix}] {line}", end="", file=output_stream)

    @property
    def model_size(self):
        return self._file_size
