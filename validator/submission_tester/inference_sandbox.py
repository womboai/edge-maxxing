import sys
import time
from multiprocessing.connection import Client
from os.path import abspath
from pathlib import Path
from subprocess import Popen, run
from typing import Generic

import bittensor as bt

from neuron import RequestT

SANDBOX_DIRECTORY = Path("/sandbox")
SETUP_INFERENCE_SANDBOX_SCRIPT = abspath(Path(__file__).parent / "setup_inference_sandbox.sh")
START_INFERENCE = abspath(SANDBOX_DIRECTORY / ".venv" / "bin" / "start_inference")
SOCKET = abspath(SANDBOX_DIRECTORY / "inferences.sock")

SANDBOX_ARGS = [
    "/bin/sudo",
    "-u",
    "sandbox",
]


class InferenceSandbox(Generic[RequestT]):
    _repository: str
    _client: Client
    _process: Popen

    def __init__(self, repository: str, revision: str):
        bt.logging.info(f"Downloading {repository} with revision {revision}")

        self._repository = repository

        run(
            [
                *SANDBOX_ARGS,
                SETUP_INFERENCE_SANDBOX_SCRIPT,
                repository,
                revision,
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        ).check_returncode()

        self._file_size = sum(file.stat().st_size for file in SANDBOX_DIRECTORY.rglob("*"))

        bt.logging.info(f"Repository {repository} had size {self._file_size}")

        self._process = Popen(
            [
                *SANDBOX_ARGS,
                START_INFERENCE
            ],
            cwd=SANDBOX_DIRECTORY,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        bt.logging.info(f"Inference process starting")
        time.sleep(60.0)

        self._check_exit()

        bt.logging.info(f"Connecting to socket")
        self._client = Client(SOCKET)

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
                *SANDBOX_ARGS,
                "rm",
                "-rf",
                str(SANDBOX_DIRECTORY / "*")
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
