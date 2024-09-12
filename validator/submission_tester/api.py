import asyncio
from contextlib import asynccontextmanager
from io import TextIOBase

from base_validator.metrics import BenchmarkState, BenchmarkResults
from fastapi import FastAPI, WebSocket, Request

from neuron import CURRENT_CONTEST, CheckpointSubmission, Key
from .benchmarker import Benchmarker

import logging


class WebSocketLogStream(TextIOBase):
    _websocket: WebSocket

    def __init__(self, websocket: WebSocket):
        super().__init__()

        self._websocket = websocket

    def write(self, text: str):
        self._websocket.send_text(text)


@asynccontextmanager
async def lifespan(_: FastAPI):
    CURRENT_CONTEST.validate()

    yield {
        "benchmarker": Benchmarker(),
    }


app = FastAPI(lifespan=lifespan)


@app.post("/start")
async def start_benchmarking(submissions: dict[Key, CheckpointSubmission], request: Request):
    benchmarker: Benchmarker = request.state.benchmarker

    await benchmarker.start_benchmarking(submissions)


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


@app.websocket("/logs")
async def stream_logs(websocket: WebSocket):
    await websocket.accept()

    handler = logging.StreamHandler(WebSocketLogStream(websocket))

    try:
        logging.root.addHandler(handler)

        while True:
            await asyncio.sleep(250)
    finally:
        # On disconnect
        logging.root.removeHandler(handler)
