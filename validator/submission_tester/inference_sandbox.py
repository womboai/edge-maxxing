import logging
import os
import sys
import time
from multiprocessing.connection import Client
from os.path import abspath
from pathlib import Path
from subprocess import Popen, run, TimeoutExpired, CalledProcessError
from typing import Generic

from neuron import RequestT, INFERENCE_SOCKET_TIMEOUT, MinerSubmissionRepositoryInfo

SETUP_INFERENCE_SANDBOX_SCRIPT = abspath(Path(__file__).parent / "setup_inference_sandbox.sh")

SANDBOX_DIRECTORY = Path("/sandbox")
BASELINE_SANDBOX_DIRECTORY = Path("/baseline-sandbox")

logger = logging.getLogger(__name__)


def sandbox_args(user: str):
    return [
        "/bin/sudo",
        "-u",
        user,
    ]


class InvalidSubmissionError(Exception):
    ...


class InferenceSandbox(Generic[RequestT]):
    _repository: MinerSubmissionRepositoryInfo

    _client: Client
    _process: Popen

    def __init__(self, repository_info: MinerSubmissionRepositoryInfo, baseline: bool):
        logger.info(f"Downloading {repository_info}")

        self._repository = repository_info

        self._baseline = baseline

        try:
            start_process = run(
                [
                    *sandbox_args(self._user),
                    SETUP_INFERENCE_SANDBOX_SCRIPT,
                    self._sandbox_directory,
                    repository_info.url,
                    repository_info.revision,
                    str(baseline).lower(),
                ],
                capture_output=True,
                encoding='utf-8',
            )
            print(start_process.stdout)
            print(start_process.stderr, file=sys.stderr)
            start_process.check_returncode()

        except CalledProcessError as e:
            if baseline:
                self.clear_sandbox()
                raise RuntimeError(f"Failed to setup baseline sandbox, cleared baseline sandbox directory: {e}")
            else:
                raise InvalidSubmissionError(f"Failed to setup sandbox: {e}")

        self._file_size = sum(file.stat().st_size for file in self._sandbox_directory.rglob("*"))

        logger.info(f"Repository {repository_info} had size {self._file_size}")

        self._process = Popen(
            [
                *sandbox_args(self._user),
                abspath(self._sandbox_directory / ".venv" / "bin" / "start_inference")
            ],
            cwd=self._sandbox_directory,
        )

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
        self._client = Client(socket_path)

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

    @property
    def model_size(self):
        return self._file_size
