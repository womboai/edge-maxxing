import time
from itertools import islice
from math import ceil

import requests
from fiber.logging_utils import get_logger
from pydantic import RootModel
from substrateinterface import Keypair

from base_validator import BenchmarkResults
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

    def __init__(self, keypair: Keypair, api: str):
        self._keypair = keypair
        self._api = api

    def start_benchmarking(self, submissions: dict[Key, ModelRepositoryInfo]):
        logger.info(f"Sending {len(submissions)} submissions for testing")

        requests.post(
            f"{self._api}/start",
            headers={
                "Content-Type": "application/json",
                **_authentication_headers(self._keypair),
            },
            data=RootModel(submissions).model_dump_json(),
        ).raise_for_status()

    def state(self):
        response = requests.get(f"{self._api}/state")
        response.raise_for_status()
        return BenchmarkResults.model_validate(response.json())

def send_submissions_to_api(apis: list[BenchmarkingApi], submissions: dict[Key, ModelRepositoryInfo]):
    iterator = iter(submissions.items())

    chunk_size = ceil(len(submissions) / len(apis))

    chunks = [
        (api, list(islice(iterator, chunk_size)))
        for api in apis
    ]

    for api, chunk in chunks:
        api.start_benchmarking(dict(chunk))
