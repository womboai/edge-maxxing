import os
import sys
import time
from multiprocessing.connection import Client
from os.path import abspath
from pathlib import Path
from subprocess import Popen, run, TimeoutExpired
from typing import Generic, cast, IO

from neuron import RequestT, bt

SETUP_INFERENCE_SANDBOX_SCRIPT = abspath(Path(__file__).parent / "setup_inference_sandbox.sh")

SANDBOX_DIRECTORY = Path("/sandbox")
BASELINE_SANDBOX_DIRECTORY = Path("/baseline-sandbox")
SOCKET_TIMEOUT = 300


def sandbox_args(user: str):
    return [
        "/bin/sudo",
        "-u",
        user,
    ]


class OutputDelegate:
    _name: str

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, item):
        return getattr(getattr(sys, self._name), item)


def delegate(name: str):
    return cast(IO[str], OutputDelegate(name))


class InferenceSandbox(Generic[RequestT]):
    _repository: str

    _client: Client
    _process: Popen

    def __init__(self, repository: str, revision: str, baseline: bool):
        bt.logging.info(f"Downloading {repository} with revision {revision}")

        self._repository = repository

        self._baseline = baseline

        run(
            [
                *sandbox_args(self._user),
                SETUP_INFERENCE_SANDBOX_SCRIPT,
                self._sandbox_directory,
                repository,
                revision,
                str(baseline).lower(),
            ],
            stdout=delegate("stdout"),
            stderr=delegate("stderr"),
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

            time.sleep(1)

            self._check_exit()
        else:
            raise RuntimeError(f"Socket file '{socket_path}' not found after {SOCKET_TIMEOUT} seconds.")

        bt.logging.info(f"Connecting to socket")
        self._client = Client(socket_path)

    @property
    def _user(self) -> str:
        return "baseline-sandbox" if self._baseline else "sandbox"

    @property
    def _sandbox_directory(self) -> Path:
        return BASELINE_SANDBOX_DIRECTORY if self._baseline else SANDBOX_DIRECTORY

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

        try:
            self._process.wait(timeout=30)

            self._check_exit()
        except TimeoutExpired:
            self._process.kill()
            bt.logging.warning(f"Forcefully killed inference process")

        self._process.__exit__(exc_type, exc_val, exc_tb)

        if not self._baseline:
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

    def __call__(self, request: RequestT):
        self._check_exit()

        data = request.model_dump_json().encode("utf-8")

        self._client.send_bytes(data)

        return self._client.recv_bytes()

    @property
    def model_size(self):
        return self._file_size
