import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from io import TextIOWrapper
from queue import Queue
from typing import Annotated, TextIO

from fastapi import FastAPI, WebSocket, Request, Header, HTTPException
from starlette import status
from substrateinterface import Keypair

from neuron import CURRENT_CONTEST, Key, ModelRepositoryInfo
from .benchmarker import Benchmarker
from base_validator import (
    API_VERSION,
    BenchmarkState,
    BenchmarkResults,
)

hotkey = os.getenv("VALIDATOR_HOTKEY_SS58_ADDRESS")
debug = int(os.getenv("VALIDATOR_DEBUG", 0)) > 0

if not hotkey:
    raise ValueError("Environment variable VALIDATOR_HOTKEY_SS58_ADDRESS was not specified")

keypair = Keypair(ss58_address=hotkey)

logs = Queue()


class LogsIO(TextIOWrapper):
    old_stdout: TextIO
    log_type: str

    def __init__(self, old_stdout: TextIO, log_type: str):
        super().__init__(old_stdout.buffer, encoding=old_stdout.encoding, errors=old_stdout.errors, newline=old_stdout.newlines)
        self.old_stdout = old_stdout
        self.log_type = log_type

    def write(self, text):
        if text.strip():
            logs.put(f"{self.log_type}: {text.strip()}")
        return self.old_stdout.write(text)

    def flush(self):
        self.old_stdout.flush()


sys.stdout = LogsIO(sys.stdout, "out")
sys.stderr = LogsIO(sys.stderr, "err")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not debug:
        CURRENT_CONTEST.validate()

    yield {
        "benchmarker": Benchmarker(),
    }


app = FastAPI(lifespan=lifespan)


def _authenticate_request(nonce: int, signature: str):
    if debug:
        return

    current_timestamp = time.time_ns()

    if current_timestamp - nonce > 2_000_000_000:
        logger.info(f"Got request with nonce {nonce}, which is {current_timestamp - nonce} nanoseconds old.")

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid nonce",
        )

    if not keypair.verify(str(nonce), signature):
        logger.info(f"Got invalid signature for nonce {nonce}: {signature}")

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid signature",
        )


@app.post("/start")
def start_benchmarking(
    submissions: dict[Key, ModelRepositoryInfo],
    x_nonce: Annotated[int, Header()],
    signature: Annotated[str, Header()],
    request: Request,
):
    _authenticate_request(x_nonce, signature)

    benchmarker: Benchmarker = request.state.benchmarker

    with benchmarker.lock:
        timestamp = time.time_ns()

        if timestamp - benchmarker.start_timestamp < 10_000_000_000:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Started recently",
            )

        benchmarker.start_timestamp = timestamp

        benchmarker.start_benchmarking(submissions)


@app.get("/state")
def state(request: Request) -> BenchmarkResults:
    benchmarker: Benchmarker = request.state.benchmarker

    benchmark_state: BenchmarkState

    if not benchmarker.benchmark_future:
        benchmark_state = BenchmarkState.NOT_STARTED
    elif benchmarker.done:
        benchmark_state = BenchmarkState.FINISHED
    else:
        benchmark_state = BenchmarkState.IN_PROGRESS

    average_benchmark_time = (
        sum(benchmarker.submission_times) / len(benchmarker.submission_times)
        if benchmarker.submission_times
        else None
    )

    return BenchmarkResults(
        state=benchmark_state,
        results=benchmarker.benchmarks,
        invalid=benchmarker.invalid,
        baseline_metrics=benchmarker.get_baseline_metrics(),
        average_benchmark_time=average_benchmark_time,
    )


@app.websocket("/logs")
async def stream_logs(
    websocket: WebSocket,
):
    nonce = int(websocket.headers["x-nonce"])
    signature = websocket.headers["signature"]

    _authenticate_request(nonce, signature)

    await websocket.accept()

    await websocket.send_json({"version": API_VERSION})

    try:
        while True:
            if not logs.empty():
                message = logs.get()
                await websocket.send_text(message)
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"WebSocket error", exc_info=e)
