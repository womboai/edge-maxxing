import sys
import time
from os.path import abspath
from pathlib import Path
from shutil import rmtree
from socket import socket, AF_UNIX, SOCK_STREAM
from subprocess import Popen, run
from sys import byteorder
from typing import Generic

import bittensor as bt

from neuron import RequestT

SANDBOX_DIRECTORY = Path("/sandbox")
SETUP_INFERENCE_SANDBOX_SCRIPT = abspath(Path(__file__).parent / "setup_inference_sandbox.sh")
START_INFERENCE = abspath(SANDBOX_DIRECTORY / ".venv" / "bin" / "start_inference")
SOCKET = abspath(SANDBOX_DIRECTORY / "inferences.sock")


class InferenceSandbox(Generic[RequestT]):
    _repository: str
    _socket: socket
    _process: Popen

    def __init__(self, repository: str, revision: str):
        bt.logging.info(f"Downloading {repository} with revision {revision}")

        self._repository = repository

        setup_result = run(
            [
                "/bin/sudo",
                "-u",
                "sandbox",
                "/bin/sh",
                SETUP_INFERENCE_SANDBOX_SCRIPT,
                repository,
                revision,
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        setup_result.check_returncode()

        self._file_size = sum(file.stat().st_size for file in SANDBOX_DIRECTORY.rglob("*"))

        bt.logging.info(f"Repository {repository} had size {self._file_size}")

        self._process = Popen(
            [
                "/bin/sudo",
                "-u",
                "sandbox",
                START_INFERENCE
            ],
            cwd=SANDBOX_DIRECTORY,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        bt.logging.info(f"Inference process starting")
        time.sleep(60.0)

        self._check_exit()

        self._socket = socket(AF_UNIX, SOCK_STREAM)

        bt.logging.info(f"Connecting to socket")
        self._socket.connect(SOCKET)

    def _check_exit(self):
        if self._process.returncode:
            raise RuntimeError(f"'{self._repository}'s inference crashed, got exit code {self._process.returncode}")

    def __enter__(self):
        self._process.__enter__()
        self._socket.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._socket.__exit__(exc_type, exc_val, exc_tb)

        self._process.terminate()
        self._process.__exit__(exc_type, exc_val, exc_tb)

        run(
            [
                "/bin/sudo",
                "-u",
                "sandbox",
                "rm -rf /sandbox/*",
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        self._check_exit()

    def __call__(self, request: RequestT):
        self._check_exit()

        data = request.model_dump_json().encode("utf-8")

        self._socket.send(len(data).to_bytes(2, byteorder))
        self._socket.send(data)

        size = int.from_bytes(self._socket.recv(4), byteorder)

        return self._socket.recv(size)

    @property
    def model_size(self):
        return self._file_size
