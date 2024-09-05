from os import popen, chmod
from pathlib import Path
from shutil import rmtree
from socket import socket, AF_UNIX, SOCK_STREAM
from sys import byteorder
from typing import Generic, ContextManager

from neuron import RequestT

SANDBOX_DIRECTORY = Path("/sandbox")
START_INFERENCE_SANDBOX_SCRIPT = Path(__file__).parent / "start_inference_sandbox.sh"
SOCKET = "/api/inferences.sock"


class InferenceSandbox(Generic[RequestT]):
    _socket: socket
    _connection: socket
    _process: ContextManager

    def __init__(self, repository: str, revision: str):
        self._file_size = sum(file.stat().st_size for file in SANDBOX_DIRECTORY.rglob("*"))

        self._process = popen(f"sudo su - sandbox -c '{START_INFERENCE_SANDBOX_SCRIPT} {repository} {revision}'")

        self._socket = socket(AF_UNIX, SOCK_STREAM)
        self._socket.bind(str(SOCKET))
        chmod(SOCKET, 0o777)

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
