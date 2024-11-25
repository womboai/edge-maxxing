import time
from collections import defaultdict
from itertools import islice
from math import ceil
from operator import itemgetter

import requests
from fiber.logging_utils import get_logger
from substrateinterface import Keypair

from base_validator import BenchmarkResults, BenchmarkingStartRequest, ApiMetadata, API_VERSION
from neuron import ModelRepositoryInfo, Key, ContestId, MinerModelInfo, CURRENT_CONTEST

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

    def __init__(self, keypair: Keypair, api: str):
        self._keypair = keypair
        self._api = api

    def start_benchmarking(self, contest_id: ContestId, submissions: dict[Key, ModelRepositoryInfo]):
        logger.info(f"Sending {len(submissions)} submissions for testing")

        data = BenchmarkingStartRequest(contest_id=contest_id, submissions=submissions)

        requests.post(
            f"{self._api}/start",
            headers={
                "Content-Type": "application/json",
                **_authentication_headers(self._keypair),
            },
            data=data.model_dump_json(),
        ).raise_for_status()

    def state(self):
        response = requests.get(f"{self._api}/state")
        response.raise_for_status()
        return BenchmarkResults.model_validate(response.json())

    def metadata(self) -> ApiMetadata:
        response = requests.get(f"{self._api}/metadata")
        response.raise_for_status()
        return ApiMetadata.model_validate(response.json())


def send_submissions_to_api(apis: list[BenchmarkingApi], submissions: dict[Key, MinerModelInfo]):
    submissions_info = {
        key: info.repository
        for key, info in submissions.items()
        if info.contest_id == CURRENT_CONTEST.id
    }

    iterator = iter(submissions_info.items())

    chunk_size = ceil(len(submissions_info) / len(apis))

    chunks = [
        (api, list(islice(iterator, chunk_size)))
        for api in apis
    ]

    for api, chunk in chunks:
        api.start_benchmarking(CURRENT_CONTEST.id, dict(chunk))