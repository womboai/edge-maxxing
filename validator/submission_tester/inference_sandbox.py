from os import popen
from pathlib import Path
from shutil import rmtree
from socket import socket, AF_UNIX, SOCK_STREAM
from sys import byteorder
from typing import Generic, ContextManager

from git import Repo

from neuron import RequestT

SANDBOX_DIRECTORY = Path("/sandbox")
START_INFERENCE_SANDBOX_SCRIPT = Path(__file__).parent / "start_inference_sandbox.sh"
SOCKET = SANDBOX_DIRECTORY / "inferences.sock"


class InferenceSandbox(Generic[RequestT]):
    _socket: socket
    _connection: socket
    _process: ContextManager

    def __init__(self, repository: str, revision: str):
        repository = Repo.clone_from(
            repository,
            "/sandbox",
            revision=revision,
            depth=1,
            recursive=True,
            no_checkout=True,
        )

        repository.git.checkout(revision)

        self._file_size = sum(file.stat().st_size for file in SANDBOX_DIRECTORY.rglob("*"))

        self._process = popen(f"su -m sandbox -c {START_INFERENCE_SANDBOX_SCRIPT}")

        self._socket = socket(AF_UNIX, SOCK_STREAM)
        self._socket.bind(str(SOCKET))

    def __enter__(self):
        self._process.__enter__()
        inference_socket = self._socket.__enter__()

        inference_socket.listen(1)
        self._connection, _ = inference_socket.accept()

        self._connection.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._connection.__exit__(exc_type, exc_val, exc_tb)
        self._socket.__exit__(exc_type, exc_val, exc_tb)
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
