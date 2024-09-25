import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from io import TextIOWrapper
from queue import Queue

from base_validator import API_VERSION
from base_validator.metrics import BenchmarkState, BenchmarkResults
from fastapi import FastAPI, WebSocket, Request
from neuron import CURRENT_CONTEST, CheckpointSubmission, Key

from .benchmarker import Benchmarker

logs = Queue()


class LogsIO(TextIOWrapper):
    old_stdout: TextIOWrapper
    log_type: str

    def __init__(self, old_stdout, log_type: str):
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
        results=benchmarker.benchmarks,
    )


@app.websocket("/logs")
async def stream_logs(websocket: WebSocket):
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
