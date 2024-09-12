import os
import sys
import time
from multiprocessing.connection import Client
from os.path import abspath
from pathlib import Path
from subprocess import Popen, run
from typing import Generic

from neuron import RequestT, bt

SETUP_INFERENCE_SANDBOX_SCRIPT = abspath(Path(__file__).parent / "setup_inference_sandbox.sh")

SANDBOX_DIRECTORY = Path("/sandbox")
BASELINE_SANDBOX_DIRECTORY = Path("/baseline-sandbox")
SOCKET_TIMEOUT = 120


def sandbox_args(user: str):
    return [
        "/bin/sudo",
        "-u",
        user,
    ]


class InferenceSandbox(Generic[RequestT]):
    _repository: str

    _user: str
    _sandbox_directory: Path

    _client: Client
    _process: Popen

    def __init__(self, repository: str, revision: str, baseline: bool):
        bt.logging.info(f"Downloading {repository} with revision {revision}")

        self._repository = repository

        self._user = "baseline-sandbox" if baseline else "sandbox"
        self._sandbox_directory = BASELINE_SANDBOX_DIRECTORY if baseline else SANDBOX_DIRECTORY

        run(
            [
                *sandbox_args(self._user),
                SETUP_INFERENCE_SANDBOX_SCRIPT,
                self._sandbox_directory,
                repository,
                revision,
                str(baseline).lower(),
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        ).check_returncode()

        self._file_size = sum(file.stat().st_size for file in self._sandbox_directory.rglob("*"))

        bt.logging.info(f"Repository {repository} had size {self._file_size}")

        self._process = Popen(
            [
                *sandbox_args(self._user),
                abspath(self._sandbox_directory / ".venv" / "bin" / "start_inference")
            ],
            cwd=self._sandbox_directory,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        bt.logging.info(f"Inference process starting")
        socket_path = abspath(self._sandbox_directory / "inferences.sock")

        for _ in range(SOCKET_TIMEOUT):
            if os.path.exists(socket_path):
                break

            self._check_exit()

            time.sleep(1)
        else:
            bt.logging.error(f"Socket file '{socket_path}' not found after {SOCKET_TIMEOUT} seconds.")

        self._check_exit()

        bt.logging.info(f"Connecting to socket")
        self._client = Client(socket_path)

    def _check_exit(self):
        if self._process.returncode:
            raise RuntimeError(f"'{self._repository}'s inference crashed, got exit code {self._process.returncode}")

    def __enter__(self):
        self._process.__enter__()
        self._client.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.__exit__(exc_type, exc_val, exc_tb)

        self._process.terminate()
        self._process.__exit__(exc_type, exc_val, exc_tb)

        run(
            [
                *sandbox_args(self._user),
                "find",
                str(self._sandbox_directory),
                "-mindepth",
                "1",
                "-delete",
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        ).check_returncode()

        self._check_exit()

    def __call__(self, request: RequestT):
        self._check_exit()

        data = request.model_dump_json().encode("utf-8")

        self._client.send_bytes(data)

        return self._client.recv_bytes()

    @property
    def model_size(self):
        return self._file_size
