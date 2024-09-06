from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.requests import Request

from base_validator.metrics import BenchmarkState, CheckpointBenchmark, MetricData


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield {
        "hotkeys": [],
    }


app = FastAPI(lifespan=lifespan)


@app.post("/start")
def start_benchmarking(submissions: dict[str, str], request: Request):
    request.state.hotkeys = list(submissions.keys())


@app.get("/state")
def state(request: Request) -> BenchmarkState:
    results = {
        hotkey: CheckpointBenchmark(
            baseline=MetricData(
                generation_time=3,
                size=2**32,
                vram_used=2**16,
                watts_used=5,
            ),
            model=MetricData(
                generation_time=2,
                size=2**24,
                vram_used=2**12,
                watts_used=4,
            ),
            similarity_score=1.0
        )
        for hotkey in request.state.hotkeys
    }

    return BenchmarkState(results=results)
