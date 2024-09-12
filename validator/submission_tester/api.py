from contextlib import asynccontextmanager

from base_validator.metrics import BenchmarkState, BenchmarkResults
from fastapi import FastAPI
from starlette.requests import Request

from neuron import CURRENT_CONTEST, CheckpointSubmission, Key
from .benchmarker import Benchmarker


@asynccontextmanager
async def lifespan(_: FastAPI):
    CURRENT_CONTEST.validate()

    yield {
        "benchmarker": Benchmarker(),
    }


app = FastAPI(lifespan=lifespan)


@app.post("/start")
def start_benchmarking(submissions: dict[Key, CheckpointSubmission], request: Request):
    benchmarker: Benchmarker = request.state.benchmarker

    benchmarker.start_benchmarking(submissions)


@app.get("/state")
def state(request: Request) -> BenchmarkResults:
    benchmarker: Benchmarker = request.state.benchmarker

    benchmark_state: BenchmarkState

    if not benchmarker.started:
        benchmark_state = BenchmarkState.NOT_STARTED
    elif benchmarker.done:
        benchmark_state = BenchmarkState.FINISHED
    else:
        benchmark_state = BenchmarkState.IN_PROGRESS

    return BenchmarkResults(
        state=benchmark_state,
        results={
            hotkey: submission
            for hotkey, submission in benchmarker.benchmarks.items()
            if submission
        },
    )
