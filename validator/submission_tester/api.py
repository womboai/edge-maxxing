import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from io import StringIO
from queue import Queue

from base_validator import API_VERSION
from base_validator.metrics import BenchmarkState, BenchmarkResults
from fastapi import FastAPI, WebSocket, Request
from neuron import CURRENT_CONTEST, CheckpointSubmission, Key

from .benchmarker import Benchmarker

logs = Queue()

class LogsIO(StringIO):
    old_stdout: StringIO

    def __init__(self, old_stdout):
        super().__init__()
        self.old_stdout = old_stdout

    def write(self, text):
        self.old_stdout.write(text)
        if text.strip():
            logs.put(text.strip())
        return super().write(text)

    def flush(self):
        self.old_stdout.flush()
        return super().flush()


sys.stdout = LogsIO(sys.stdout)
sys.stderr = LogsIO(sys.stderr)
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
        results={
            hotkey: submission
            for hotkey, submission in benchmarker.benchmarks.items()
            if submission
        },
    )


@app.websocket("/logs")
async def stream_logs(websocket: WebSocket):
    await websocket.accept()

    await websocket.send_json({"version": API_VERSION})

    try:
        while True:
            if not logs.empty():
                message = logs.get()
                await websocket.send_text(f"[API] - {message}")
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
