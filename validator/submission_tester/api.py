import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Request, Header, HTTPException
from starlette import status
from substrateinterface import Keypair

from base_validator import (
    API_VERSION,
    BenchmarkState,
    BenchmarkResults,
    AutoUpdater,
    init_open_telemetry_logging,
)
from neuron import CURRENT_CONTEST, Key, ModelRepositoryInfo
from .benchmarker import Benchmarker

hotkey = os.getenv("VALIDATOR_HOTKEY_SS58_ADDRESS")
debug = int(os.getenv("VALIDATOR_DEBUG") or 0) > 0

if not hotkey:
    raise ValueError("Environment variable VALIDATOR_HOTKEY_SS58_ADDRESS was not specified")

keypair = Keypair(ss58_address=hotkey)

init_open_telemetry_logging({
    "neuron.hotkey": hotkey,
    "api.version": API_VERSION,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    AutoUpdater()
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

        if timestamp - benchmarker.start_timestamp < 120_000_000_000:
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

    if not benchmarker.thread:
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
