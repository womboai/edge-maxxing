import asyncio
import sys
from asyncio import Future, AbstractEventLoop
from contextlib import asynccontextmanager
from io import TextIOBase
from typing import TextIO

from base_validator.metrics import BenchmarkState, BenchmarkResults
from fastapi import FastAPI, WebSocket, Request

from neuron import CURRENT_CONTEST, CheckpointSubmission, Key
from .benchmarker import Benchmarker


_DELEGATION_NAMES = [
    "closed",
    "_checkClosed",
    "_checkReadable",
    "_checkSeekable",
    "_checkWritable",
    "close",
    "detach",
    "fileno",
    "isatty",
    "next",
    "read",
    "readable",
    "readline",
    "readlines",
    "seek",
    "seekable",
    "tell",
    "truncate",
    "writable",
    "__iter__",
]

async def send_data(websocket: WebSocket, data: list[str]):
    for line in data:
        await websocket.send_text(line)


class WebSocketLogStream(TextIOBase):
    _websocket: WebSocket
    _log_type: str
    _delegate: TextIO

    _data: list[str]
    _futures: list[Future[None]]

    _loop: AbstractEventLoop

    def __init__(self, websocket: WebSocket, log_type: str, delegate: TextIO, loop: AbstractEventLoop):
        super().__init__()

        self._websocket = websocket
        self._log_type = log_type
        self._delegate = delegate

        for name in _DELEGATION_NAMES:
            if hasattr(self._delegate, name):
                setattr(self, name, getattr(self._delegate, name))

        self._data = []
        self._futures = []

        self._loop = loop

    def __enter__(self):
        self._delegate.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._delegate.__exit__(exc_type, exc_val, exc_tb)

    def __await__(self):
        yield from asyncio.gather(*self._futures).__await__()

        self._futures.clear()

    def __getattr__(self, item):
        attribute_owner = self if hasattr(self, item) else self._delegate

        return getattr(attribute_owner, item)

    def flush(self):
        self._delegate.flush()

        coroutine = send_data(self._websocket, self._data.copy())
        future = asyncio.ensure_future(coroutine, loop=self._loop)

        self._futures.append(future)

        self._data.clear()

    def write(self, text: str):
        count = self._delegate.write(text)

        self._data.append(f"{self._log_type}:{text}")

        return count


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

    loop = asyncio.get_running_loop()

    old_out = sys.stdout
    old_err = sys.stderr

    out = WebSocketLogStream(websocket, "out", sys.stdout, loop)
    err = WebSocketLogStream(websocket, "err", sys.stderr, loop)

    try:
        sys.stdout = out
        sys.stderr = err

        while True:
            await asyncio.sleep(500)

            await asyncio.gather(out, err)
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


async def main():
    print(WebSocketLogStream(None, "out", sys.stdout, asyncio.get_running_loop()).__next__)


if __name__ == '__main__':
    asyncio.run(main())
