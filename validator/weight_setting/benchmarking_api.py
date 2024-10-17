import asyncio
import json
import sys
import time
from asyncio import Task
from collections.abc import Callable, Awaitable

from aiohttp import ClientSession
from base_validator import API_VERSION, BenchmarkResults
from fiber.logging_utils import get_logger
from pydantic import RootModel
from substrateinterface import Keypair
from websockets import connect, ConnectionClosedError

from neuron import ModelRepositoryInfo, Key

logger = get_logger(__name__)


def _authentication_headers(keypair: Keypair):
    nonce = str(time.time_ns())

    signature = f"0x{keypair.sign(nonce).hex()}"

    return {
        "X-Nonce": nonce,
        "Signature": signature,
    }


class BenchmarkingApi:
    _keypair: Keypair
    _api: str
    _index: int

    _task: Task

    _stream_logs: Callable[[], Task]

    _session: ClientSession

    def __init__(
        self,
        keypair: Keypair,
        api: str,
        index: int,

        stream_logs: Callable[[], Task],
    ):
        self._keypair = keypair
        self._api = api
        self._index = index

        self._task = stream_logs()
        self._stream_logs = stream_logs

        self._session = ClientSession()

    async def start_benchmarking(self, submissions: dict[Key, ModelRepositoryInfo]):
        if self._task.done() and self._task.exception():
            logger.error("Error in log streaming", exc_info=self._task.exception())

            self._task = self._stream_logs()
        elif self._task.cancelled():
            logger.error("Log streaming task was cancelled, restarting it")

            self._task = self._stream_logs()

        logger.info(f"Sending {submissions} for testing")

        request = self._session.post(
            f"{self._api}/start",
            headers={
                "Content-Type": "application/json",
                **_authentication_headers(self._keypair),
            },
            data=RootModel(submissions).model_dump_json(),
        )

        async with request as state_response:
            state_response.raise_for_status()

    async def state(self):
        async with self._session.get(f"{self._api}/state") as state_response:
            state_response.raise_for_status()

            return BenchmarkResults.model_validate(await state_response.json())

    async def close(self):
        self._task.cancel()
        await self._session.close()


class BenchmarkingApiContextManager(Awaitable[BenchmarkingApi]):
    _keypair: Keypair
    _api: str
    _index: int

    def __init__(self, keypair: Keypair, api: str, index: int):
        self._keypair = keypair
        self._api = api
        self._index = index

    async def _connect_to_api(self):
        url = self._api.replace("http", "ws")

        websocket = await connect(f"{url}/logs", extra_headers=_authentication_headers(self._keypair))

        try:
            version = json.loads(await websocket.recv())["version"]
        except:
            raise RuntimeError("Validator API out of date")

        if version != API_VERSION:
            raise RuntimeError(
                f"Validator API has mismatched version, received {version} but expected {API_VERSION}"
            )

        return websocket

    def _stream_logs(self):
        return asyncio.create_task(self._api_logs())

    async def _create_connection(self):
        return BenchmarkingApi(
            self._keypair,
            self._api,
            self._index,

            self._stream_logs,
        )

    async def _api_logs(self):
        websocket = await self._connect_to_api()

        try:
            while True:
                try:
                    async for line in websocket:
                        output = sys.stderr if line.startswith("err:") else sys.stdout

                        print(f"[API - {self._index + 1}] - {line[4:]}", file=output)
                except ConnectionClosedError:
                    websocket = await self._connect_to_api()
        finally:
            await websocket.close()

    def __await__(self):
        return self._create_connection().__await__()


benchmarking_api = BenchmarkingApiContextManager
