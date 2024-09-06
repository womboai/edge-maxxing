import sys
import time
from os import chmod
from os.path import abspath
from pathlib import Path
from shutil import rmtree
from socket import socket, AF_UNIX, SOCK_STREAM
from subprocess import Popen
from sys import byteorder
from typing import Generic

import bittensor as bt

from neuron import RequestT

SANDBOX_DIRECTORY = Path("/sandbox")
START_INFERENCE_SANDBOX_SCRIPT = abspath(Path(__file__).parent / "start_inference_sandbox.sh")
SOCKET = "/api/inferences.sock"


class InferenceSandbox(Generic[RequestT]):
    _repository: str
    _socket: socket
    _process: Popen

    def __init__(self, repository: str, revision: str):
        bt.logging.info(f"Downloading {repository} with revision {revision}")

        self._repository = repository

        self._process = Popen(
            [
                "/bin/sudo",
                "-i",
                "-u",
                "sandbox",
                "/bin/sh",
                START_INFERENCE_SANDBOX_SCRIPT,
                repository,
                revision,
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        bt.logging.info(f"Inference process starting")
        time.sleep(10.0)

        self._check_exit()

        bt.logging.info(f"Binding to inference socket")
        self._socket = socket(AF_UNIX, SOCK_STREAM)
        self._socket.bind(str(SOCKET))
        chmod(SOCKET, 0o777)

        bt.logging.info(f"Waiting for inference container to be ready")
        self._socket.settimeout(60.0)
        self._socket.connect(SOCKET)

        self._check_exit()

        self._file_size = sum(file.stat().st_size for file in SANDBOX_DIRECTORY.rglob("*"))

        bt.logging.info(f"Repository {repository} had size {self._file_size}")

    def _check_exit(self):
        if self._process.returncode:
            raise RuntimeError(f"Failed to setup {self._repository}, got exit code {self._process.returncode}")

    def __enter__(self):
        self._process.__enter__()
        self._socket.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._socket.__exit__(exc_type, exc_val, exc_tb)

        self._process.terminate()
        self._process.__exit__(exc_type, exc_val, exc_tb)

        rmtree("/sandbox")

    def __call__(self, request: RequestT):
        data = request.model_dump_json().encode("utf-8")

        self._socket.send(len(data).to_bytes(2, byteorder))
        self._socket.send(data)

        size = int.from_bytes(self._socket.recv(4), byteorder)

        return self._socket.recv(size)

    @property
    def model_size(self):
        return self._file_size
