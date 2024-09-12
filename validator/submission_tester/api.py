import asyncio
from asyncio import Future, AbstractEventLoop
from contextlib import asynccontextmanager
from io import TextIOBase, StringIO

from base_validator.metrics import BenchmarkState, BenchmarkResults
from fastapi import FastAPI, WebSocket, Request

from neuron import CURRENT_CONTEST, CheckpointSubmission, Key
from .benchmarker import Benchmarker

import logging


class WebSocketLogStream(TextIOBase):
    _websocket: WebSocket
    _buffer: StringIO

    _futures: list[Future[None]]
    _loop: AbstractEventLoop

    def __init__(self, websocket: WebSocket, loop: AbstractEventLoop):
        super().__init__()

        self._websocket = websocket
        self._buffer = StringIO()

        self._loop = loop

    def __enter__(self):
        self._buffer.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._buffer.__exit__(exc_type, exc_val, exc_tb)

    def __await__(self):
        yield from asyncio.gather(*self._futures).__await__()

        self._futures.clear()

    def flush(self):
        buffered_text = self._buffer.getvalue()

        self._futures.append(asyncio.ensure_future(self._websocket.send_text(buffered_text), loop=self._loop))

        self._buffer.truncate(0)
        self._buffer.seek(0)

    def write(self, text: str):
        self._buffer.write(text)


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

    with WebSocketLogStream(websocket, asyncio.get_running_loop()) as stream:
        handler = logging.StreamHandler(stream)

        try:
            logging.root.addHandler(handler)

            while True:
                await asyncio.sleep(500)

                await stream
        finally:
            # On disconnect
            logging.root.removeHandler(handler)
