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


def send_submissions_to_api(all_apis: list[BenchmarkingApi], submissions: dict[Key, MinerModelInfo]):
    submissions_by_contest: dict[ContestId, dict[Key, ModelRepositoryInfo]] = defaultdict(lambda: {})

    for key, info in submissions.items():
        if info.contest_id != CURRENT_CONTEST:
            continue  # TODO: Remove once multi-competition support is added
        submissions_by_contest[info.contest_id][key] = info.repository

    contest_api_assignment: dict[ContestId, list[BenchmarkingApi]] = defaultdict(lambda: [])

    for api in all_apis:
        metadata = api.metadata()
        if metadata.version != API_VERSION:
            raise ValueError(f"API version mismatch, expected {API_VERSION}, got {metadata.version}")

        if sum(len(contest_api_assignment[contest_id]) for contest_id in metadata.compatible_contests) == 0:
            compatible_contests = list(filter(submissions_by_contest.__contains__, metadata.compatible_contests))

            if compatible_contests:
                contest_api_assignment[compatible_contests[0]].append(api)
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

        for api, chunk in chunks:
            api.start_benchmarking(contest_id, dict(chunk))
