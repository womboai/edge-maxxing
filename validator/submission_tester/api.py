import os
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from time import time_ns
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.params import Header, Body
from fiber.logging_utils import get_logger
from starlette.requests import Request
from starlette.status import HTTP_403_FORBIDDEN, HTTP_400_BAD_REQUEST, HTTP_429_TOO_MANY_REQUESTS
from substrateinterface import Keypair

from base.contest import find_compatible_contests, find_contest, CONTESTS
from base_validator.api_data import BenchmarkingStartRequest, BenchmarkingResults, ApiMetadata, BenchmarkingInitializeRequest
from base_validator.auto_updater import AutoUpdater
from base_validator.telemetry import init_open_telemetry_logging
from testing.benchmarker import Benchmarker

hotkey = os.getenv("VALIDATOR_HOTKEY_SS58_ADDRESS")
debug = int(os.getenv("VALIDATOR_DEBUG") or 0) > 0

if not hotkey:
    raise ValueError("Environment variable VALIDATOR_HOTKEY_SS58_ADDRESS was not specified")

keypair = Keypair(ss58_address=hotkey)

api_version = version("edge-maxxing-validator")

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    AutoUpdater()

    compatible_contests = find_compatible_contests() if not debug else [contest.id for contest in CONTESTS]
    if not compatible_contests:
        raise RuntimeError("Device is not compatible with any contests")

    yield {
        "benchmarker": Benchmarker(
            sandbox_directory=Path("/sandbox"),
            sandbox_args=["/bin/sudo", "-u", "sandbox"]
        ),
        "compatible_contests": compatible_contests,
    }

app = FastAPI(lifespan=lifespan)

def _authenticate_request(nonce: int, signature: str):
    if debug:
        return

    current_timestamp = time_ns()

    if current_timestamp - nonce > 2_000_000_000:
        logger.info(f"Got request with nonce {nonce}, which is {current_timestamp - nonce} nanoseconds old.")

        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid nonce",
        )

    if not keypair.verify(str(nonce), signature):
        logger.info(f"Got invalid signature for nonce {nonce}: {signature}")

        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid signature",
        )

@app.post("/start")
def start(
    start_request: Annotated[BenchmarkingStartRequest, Body()],
    x_nonce: Annotated[int, Header()],
    signature: Annotated[str, Header()],
    request: Request,
):
    _authenticate_request(x_nonce, signature)

    benchmarker: Benchmarker = request.state.benchmarker

    contest = find_contest(start_request.contest_id)
    if not debug and not contest.device.is_compatible():
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=str("API is not compatible with contest"),
        )

    timestamp = time_ns()
    if timestamp - benchmarker.start_timestamp < 10_000_000_000:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="Started recently",
        )

    benchmarker.start_timestamp = timestamp
    benchmarker.start_benchmarking(contest, start_request.submissions)

@app.get("/state")
def state(request: Request) -> BenchmarkingResults:
    benchmarker: Benchmarker = request.state.benchmarker

    average_benchmark_time = (
        sum(benchmarker.submission_times) / len(benchmarker.submission_times)
        if benchmarker.submission_times
        else None
    )

    return BenchmarkingResults(
        state=benchmarker.state,
        benchmarks=benchmarker.benchmarks,
        invalid_submissions=benchmarker.invalid_submissions,
        average_benchmark_time=average_benchmark_time,
    )

@app.get("/metadata")
def metadata(request: Request) -> ApiMetadata:
    return ApiMetadata(
        version=api_version,
        compatible_contests=request.state.compatible_contests,
    )

@app.post("/initialize")
def initialize(
    init_request: Annotated[BenchmarkingInitializeRequest, Body()],
    x_nonce: Annotated[int, Header()],
    signature: Annotated[str, Header()],
):
    _authenticate_request(x_nonce, signature)

    init_open_telemetry_logging({
        "neuron.uid": init_request.uid,
        "neuron.signature": init_request.signature,
        "subtensor.chain_endpoint": init_request.substrate_url,
        "api.version": api_version,
    })
