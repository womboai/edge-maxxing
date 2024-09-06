import time
from os import chmod
from pathlib import Path
from shutil import rmtree
from socket import socket, AF_UNIX, SOCK_STREAM
from subprocess import Popen, PIPE
from sys import byteorder
from typing import Generic

import bittensor as bt

from neuron import RequestT

SANDBOX_DIRECTORY = Path("/sandbox")
START_INFERENCE_SANDBOX_SCRIPT = Path(__file__).parent / "start_inference_sandbox.sh"
SOCKET = "/api/inferences.sock"


class InferenceSandbox(Generic[RequestT]):
    _repository: str
    _socket: socket
    _connection: socket
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
            stdout=PIPE,
            stderr=PIPE,
        )

        time.sleep(10.0)

        self._check_exit()

        self._socket = socket(AF_UNIX, SOCK_STREAM)
        self._socket.bind(str(SOCKET))
        chmod(SOCKET, 0o777)

        self._socket.listen(1)
        self._connection, _ = self._socket.accept()

        self._connection.settimeout(60.0)
        marker = self._connection.recv(1)

        self._check_exit()

        if marker == b'\xFF':
            # Ready
            self._file_size = sum(file.stat().st_size for file in SANDBOX_DIRECTORY.rglob("*"))

            bt.logging.info(f"Repository {repository} had size {self._file_size}")
        else:
            raise RuntimeError(
                f"Repository {repository} is invalid, did not receive proper READY marker, got {marker} instead"
            )

    def _check_exit(self):
        if self._process.returncode:
            raise RuntimeError(f"Failed to setup {self._repository}, got exit code {self._process.returncode}")

    def __enter__(self):
        self._process.__enter__()
        self._socket.__enter__()
        self._connection.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._connection.__exit__(exc_type, exc_val, exc_tb)
        self._socket.__exit__(exc_type, exc_val, exc_tb)

        self._process.terminate()
        self._process.__exit__(exc_type, exc_val, exc_tb)

        rmtree("/sandbox")

    def __call__(self, request: RequestT):
        data = request.model_dump_json().encode("utf-8")

        self._connection.send(len(data).to_bytes(2, byteorder))
        self._connection.send(data)

        size = int.from_bytes(self._connection.recv(4), byteorder)

        return self._connection.recv(size)

    @property
    def model_size(self):
        return self._file_size
