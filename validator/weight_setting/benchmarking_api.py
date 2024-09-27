import asyncio
import json
import sys
import time
from asyncio import Task
from collections.abc import Callable, Awaitable

from aiohttp import ClientSession
from fiber.logging_utils import get_logger
from pydantic import RootModel
from substrateinterface import Keypair
from websockets import connect, WebSocketClientProtocol, ConnectionClosedError

from neuron import Key, CheckpointSubmission
from base_validator import API_VERSION
from base_validator.metrics import BenchmarkResults

logger = get_logger(__name__)


class BenchmarkingApi:
    _keypair: Keypair
    _api: str
    _index: int

    _websocket: WebSocketClientProtocol
    _task: Task

    _stream_logs: Callable[[], Awaitable[tuple[WebSocketClientProtocol, Task]]]

    _session: ClientSession

    def __init__(
        self,
        keypair: Keypair,
        api: str,
        index: int,

        websocket: WebSocketClientProtocol,
        task: Task,

        stream_logs: Callable[[], Awaitable[tuple[WebSocketClientProtocol, Task]]],
    ):
        self._keypair = keypair
        self._api = api
        self._index = index

        self._websocket = websocket
        self._task = task

        self._stream_logs = stream_logs

        self._session = ClientSession()

    async def start_benchmarking(self, submissions: dict[Key, CheckpointSubmission]):
        if self._task.done() and self._task.exception():
            logger.error("Error in log streaming", exc_info=self._task.exception())

            self._websocket, self._task = await self._stream_logs()
        elif self._task.cancelled():
            logger.error("Log streaming task was cancelled, restarting it")

            self._websocket, self._task = await self._stream_logs()

        logger.info(f"Sending {submissions} for testing")

        submissions_json = RootModel[dict[Key, CheckpointSubmission]](submissions).model_dump_json()

        nonce = str(time.time_ns())

        signature = f"0x{self._keypair.sign(nonce).hex()}"

        request = self._session.post(
            f"{self._api}/start",
            headers={
                "Content-Type": "application/json",
                "X-Nonce": nonce,
                "Signature": signature,
            },
            data=submissions_json,
        )

        async with request as state_response:
            state_response.raise_for_status()

    async def state(self):
        async with self._session.get(f"{self._api}/state") as state_response:
            state_response.raise_for_status()

            return BenchmarkResults.model_validate(state_response.json())

    async def close(self):
        self._task.cancel()

        await self._websocket.close()


class BenchmarkingApiContextManager(Awaitable):
    _keypair: Keypair
    _api: str
    _index: int

    def __init__(self, keypair: Keypair, api: str, index: int):
        self._keypair = keypair
        self._api = api
        self._index = index

    async def _connect_to_api(self):
        url = self._api.replace("http", "ws")

        websocket = await connect(f"{url}/logs")

        try:
            version = json.loads(await websocket.recv())["version"]
        except:
            raise RuntimeError("Validator API out of date")

        if version != API_VERSION:
            raise RuntimeError(
                f"Validator API has mismatched version, received {version} but expected {API_VERSION}"
            )

        return websocket

    async def _stream_logs(self):
        websocket = await self._connect_to_api()

        task = asyncio.create_task(self._api_logs())

        return websocket, task

    async def _create_connection(self):
        websocket, task = await self._stream_logs()

        return BenchmarkingApi(
            self._keypair,
            self._api,
            self._index,

            websocket,
            task,

            self._stream_logs,
        )

    async def _api_logs(self):
        while True:
            try:
                async for line in self._websocket:
                    output = sys.stderr if line.startswith("err:") else sys.stdout

                    print(f"[API - {self._index + 1}] - {line[4:]}", file=output)
            except ConnectionClosedError:
                self._websocket = await self._connect_to_api()

    def __await__(self):
        return self._create_connection().__await__()


benchmarking_api = BenchmarkingApiContextManager
