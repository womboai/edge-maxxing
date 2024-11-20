import asyncio
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, Executor, ThreadPoolExecutor, CancelledError
from itertools import islice
from math import ceil
from operator import itemgetter

from aiohttp import ClientSession
from fiber.logging_utils import get_logger
from pydantic import RootModel
from substrateinterface import Keypair
from websockets.protocol import State
from websockets.sync.client import connect

from base_validator import API_VERSION, BenchmarkResults, ApiMetadata
from neuron import ModelRepositoryInfo, Key, ContestId, MinerModelInfo

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

    async def start_benchmarking(self, submissions: dict[Key, ModelRepositoryInfo]):
        self.check_log_stream()

        if not self._session:
            self._session = ClientSession()

        logger.info(f"Sending {len(submissions)} submissions for testing")

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

    async def state(self) -> BenchmarkResults:
        self.check_log_stream()

        if not self._session:
            self._session = ClientSession()

        async with self._session.get(f"{self._api}/state") as state_response:
            state_response.raise_for_status()

            return BenchmarkResults.model_validate(await state_response.json())

    async def metadata(self) -> ApiMetadata:
        self.check_log_stream()

        if not self._session:
            self._session = ClientSession()

        async with self._session.get(f"{self._api}/metadata") as metadata_response:
            metadata_response.raise_for_status()

            return ApiMetadata.model_validate(await metadata_response.json())

    async def close(self):
        self._future.cancel()

        if self._session:
            await self._session.close()


class BenchmarkingApiContextManager:
    _keypair: Keypair
    _api: str
    _index: int

    _executor: Executor

    _version: int | None = None
    _compatible_contests: list[ContestId] = []

    def __init__(self, keypair: Keypair, api: str, index: int):
        self._keypair = keypair
        self._api = api
        self._index = index
        self._executor = ThreadPoolExecutor(1)

    def _connect_to_api(self):
        name = f"API - {self._index + 1}"
        logger.info(f"Connecting to {name}")
        url = self._api.replace("http", "ws")

        websocket = connect(f"{url}/logs", additional_headers=_authentication_headers(self._keypair))

        logger.info(f"Connected to {name}")
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


async def send_submissions_to_api(all_apis: list[BenchmarkingApi], submissions: dict[Key, MinerModelInfo]):
    submissions_by_contest: dict[ContestId, dict[Key, ModelRepositoryInfo]] = defaultdict(lambda: {})

    for key, info in submissions.items():
        submissions_by_contest[info.contest_id][key] = info.repository

    contest_api_assignment: dict[ContestId, list[BenchmarkingApi]] = defaultdict(lambda: [])

    for api in all_apis:
        metadata = await api.metadata()
        if metadata.version != API_VERSION:
            raise ValueError(f"API version mismatch, expected {API_VERSION}, got {metadata.version}")

        if sum(len(contest_api_assignment[contest_id]) for contest_id in metadata.compatible_contests) == 0:
            contest_id = next(filter(submissions_by_contest.__contains__, metadata.compatible_contests))

            contest_api_assignment[contest_id].append(api)
        else:
            assignment_counts = [
                (contest_id, len(contest_api_assignment[contest_id]))
                for contest_id in metadata.compatible_contests
                if contest_id in submissions_by_contest
            ]

            lowest_contest_id = min(assignment_counts, key=itemgetter(1))[0]

            contest_api_assignment[lowest_contest_id].append(api)

    for contest_id, apis in contest_api_assignment.items():
        if contest_id not in submissions_by_contest:
            raise RuntimeError(f"No API compatible with contest type {contest_id}")

        contest_submissions = submissions_by_contest[contest_id]

        iterator = iter(contest_submissions.items())

        chunk_size = ceil(len(contest_submissions) / len(apis))

        chunks = [
            (api, list(islice(iterator, chunk_size)))
            for api in apis
        ]

        await asyncio.gather(
            *[
                api.start_benchmarking(dict(chunk))
                for api, chunk in chunks
            ],
        )


benchmarking_api = BenchmarkingApiContextManager
