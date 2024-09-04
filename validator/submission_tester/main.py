from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from starlette.requests import Request

from submission_tester.benchmarker import Benchmarker
from base_validator.metrics import CheckpointBenchmark


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield {
        "benchmarker": Benchmarker(),
    }


app = FastAPI(lifespan=lifespan)


@app.post("/start")
def start_benchmarking(submissions: list[tuple[str, str]], background_tasks: BackgroundTasks, request: Request):
    benchmarker: Benchmarker = request.app.state["benchmarker"]

    background_tasks.add_task(
        benchmarker.start_benchmarking,
        submissions,
    )


@app.get("/state")
def state(request: Request) -> dict[str, CheckpointBenchmark] | None:
    benchmarker: Benchmarker = request.app.state["benchmarker"]

    if not benchmarker.done:
        return None

    return benchmarker.metrics
