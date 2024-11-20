import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future, Executor, ThreadPoolExecutor, CancelledError

from aiohttp import ClientSession
from base_validator import API_VERSION, BenchmarkResults
from fiber.logging_utils import get_logger
from pydantic import RootModel
from substrateinterface import Keypair
from websockets.protocol import State
from websockets.sync.client import connect

from neuron import ModelRepositoryInfo, Key, ContestId

RETRY_TIME = 60

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

    _future: Future

    _stream_logs: Callable[[], Future]

    _session: ClientSession | None

    def __init__(
        self,
        keypair: Keypair,
        api: str,
        index: int,

        stream_logs: Callable[[], Future],
    ):
        self._keypair = keypair
        self._api = api
        self._index = index

        self._future = stream_logs()
        self._stream_logs = stream_logs

        self._session = None

    def check_log_stream(self):
        if self._future.done():
            try:
                self._future.result()
            except Exception as e:
                logger.error("Error in log streaming", exc_info=e)

            self._future = self._stream_logs()
        elif self._future.cancelled():
            logger.error("Log streaming future was cancelled, restarting it")

            self._future = self._stream_logs()

    async def start_benchmarking(self, contest_submissions: dict[ContestId, dict[Key, ModelRepositoryInfo]]):
        self.check_log_stream()

        if not self._session:
            self._session = ClientSession()

        request = self._session.post(
            f"{self._api}/start",
            headers={
                "Content-Type": "application/json",
                **_authentication_headers(self._keypair),
            },
            data=RootModel(contest_submissions).model_dump_json(),
        )

        async with request as state_response:
            state_response.raise_for_status()

    async def state(self):
        self.check_log_stream()

        if not self._session:
            self._session = ClientSession()

        async with self._session.get(f"{self._api}/state") as state_response:
            state_response.raise_for_status()

            return BenchmarkResults.model_validate(await state_response.json())

    async def close(self):
        self._future.cancel()

        if self._session:
            await self._session.close()


class BenchmarkingApiContextManager:
    _keypair: Keypair
    _api: str
    _index: int

    _executor: Executor

    def __init__(self, keypair: Keypair, api: str, index: int):
        self._keypair = keypair
        self._api = api
        self._index = index
        self._executor = ThreadPoolExecutor(1)

    def _connect_to_api(self):
        name = f"API - {self._index + 1}"
        while True:
            logger.info(f"Connecting to {name}")
            url = self._api.replace("http", "ws")

            websocket = connect(f"{url}/logs", additional_headers=_authentication_headers(self._keypair))

            try:
                version = json.loads(websocket.recv())["version"]
            except:
                logger.error(f"Validator {name} out of date. Retrying in {RETRY_TIME} seconds")
                websocket.close()
                time.sleep(RETRY_TIME)
                continue

            if version != API_VERSION:
                logger.error(f"Validator {name} has mismatched version, received {version} but expected {API_VERSION}. Retrying in {RETRY_TIME} seconds")
                websocket.close()
                time.sleep(RETRY_TIME)
                continue

            logger.info(f"Connected to {name} with version {version}")
            return websocket

    def _stream_logs(self):
        return self._executor.submit(self._api_logs)

    def _api_logs(self):
        websocket = self._connect_to_api()

        try:
            while True:
                try:
                    for line in websocket:
                        output = sys.stderr if line.startswith("err:") else sys.stdout

                        print(f"[API - {self._index + 1}] - {line[4:]}", file=output)

                        websocket.ping()
                except (TimeoutError, CancelledError):
                    raise
                except Exception:
                    if websocket.protocol.state is State.CLOSED or websocket.protocol.state is State.CLOSING:
                        logger.error(f"Disconnected from API-{self._index + 1}'s logs, reconnecting", exc_info=True)

                        websocket = self._connect_to_api()
                    else:
                        logger.error(f"Error occurred from API-{self._index + 1}'s logs", exc_info=True)
        finally:
            websocket.close()

    def build(self):
        return BenchmarkingApi(
            self._keypair,
            self._api,
            self._index,

            self._stream_logs,
        )


benchmarking_api = BenchmarkingApiContextManager
