from collections import defaultdict
from itertools import islice
from math import ceil
from operator import itemgetter
from time import time_ns

import requests
from fiber.logging_utils import get_logger
from substrateinterface import Keypair

from base.checkpoint import Key, Uid, Submissions
from base.contest import ContestId, RepositoryInfo
from base_validator.api_data import BenchmarkingStartRequest, ApiMetadata, BenchmarkingResults, BenchmarkingInitializeRequest

logger = get_logger(__name__)

class BenchmarkingApi:
    _api: str
    _keypair: Keypair

    def __init__(self, api: str, keypair: Keypair):
        self._api = api
        self._keypair = keypair

    def initialize(self, uid: Uid, signature: str, substrate_url: str):
        requests.post(
            f"{self._api}/initialize",
            headers={
                "Content-Type": "application/json",
                **_authentication_headers(self._keypair),
            },
            data=BenchmarkingInitializeRequest(
                uid=uid,
                signature=signature,
                substrate_url=substrate_url,
            ).model_dump_json(),

        ).raise_for_status()

    def start(self, contest_id: ContestId, submissions: dict[Key, RepositoryInfo]):
        logger.info(f"Sending {len(submissions)} submissions for {contest_id.name}")

        requests.post(
            f"{self._api}/start",
            headers={
                "Content-Type": "application/json",
                **_authentication_headers(self._keypair),
            },
            data=BenchmarkingStartRequest(
                contest_id=contest_id,
                submissions=submissions,
            ).model_dump_json(),
        ).raise_for_status()

    def metadata(self) -> ApiMetadata:
        response = requests.get(f"{self._api}/metadata")
        response.raise_for_status()
        return ApiMetadata.model_validate(response.json())

    def results(self) -> BenchmarkingResults:
        response = requests.get(f"{self._api}/state")
        response.raise_for_status()
        return BenchmarkingResults.model_validate(response.json())

def _authentication_headers(keypair: Keypair):
    nonce = str(time_ns())

    signature = f"0x{keypair.sign(nonce).hex()}"

    return {
        "X-Nonce": nonce,
        "Signature": signature,
    }

def send_submissions_to_api(version: str, all_apis: list[BenchmarkingApi], submissions: Submissions):
    submissions_by_contest: dict[ContestId, dict[Key, RepositoryInfo]] = defaultdict(lambda: {})

    for key, info in submissions.items():
        submissions_by_contest[info.contest_id][key] = info.repository

    contest_api_assignment: dict[ContestId, list[BenchmarkingApi]] = defaultdict(lambda: [])

    for api in all_apis:
        metadata = api.metadata()
        if metadata.version != version:
            raise ValueError(f"API version mismatch, expected {version}, got {metadata.version}")

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
            api.start(contest_id, dict(chunk))
