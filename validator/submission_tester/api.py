from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from starlette.requests import Request

from submission_tester.benchmarker import Benchmarker

from base_validator.metrics import BenchmarkState

from neuron import CURRENT_CONTEST


@asynccontextmanager
async def lifespan(_: FastAPI):
    CURRENT_CONTEST.validate()

    yield {
        "benchmarker": Benchmarker(),
    }


app = FastAPI(lifespan=lifespan)


@app.post("/start")
def start_benchmarking(submissions: dict[str, str], background_tasks: BackgroundTasks, request: Request):
    benchmarker: Benchmarker = request.state.benchmarker

    background_tasks.add_task(
        benchmarker.start_benchmarking,
        submissions,
    )


@app.get("/state")
def state(request: Request) -> BenchmarkState:
    benchmarker: Benchmarker = request.state.benchmarker

    if not benchmarker.done:
        return BenchmarkState(results=None)

    return BenchmarkState(results=benchmarker.metrics)
