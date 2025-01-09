import os

import requests
from cachetools import TTLCache, cached
from pydantic import RootModel, BaseModel

from pipelines import TextToImageRequest
from .checkpoint import Key
from .contest import ContestId, MetricType

INPUTS_ENDPOINT = os.getenv("INPUTS_ENDPOINT", "https://edge-inputs.api.wombo.ai")


class InputsState(BaseModel):
    benchmarks_version: int
    delayed_weight_setting: bool
    winner_percentage: float
    active_contests: dict[ContestId, dict[MetricType, int]]

    def get_metric_weights(self, contest_id: ContestId) -> dict[MetricType, int]:
        if contest_id not in self.active_contests:
            raise ValueError(f"Contest {contest_id} is not active")

        return self.active_contests[contest_id]

    def get_active_contests(self) -> set[ContestId]:
        return set(self.active_contests.keys())


class Blacklist(BaseModel):
    coldkeys: set[Key]
    hotkeys: set[Key]
    dependencies: set[str]

    def is_blacklisted(self, hotkey: Key, coldkey: Key) -> bool:
        if hotkey in self.hotkeys or coldkey in self.coldkeys:
            return True

        return any(
            submission.hotkey == hotkey
            for submission in get_duplicate_submissions()
        )


class DuplicateSubmission(BaseModel):
    hotkey: Key
    copy_of: Key
    url: str


def random_inputs() -> list[TextToImageRequest]:
    response = requests.get(
        f"{INPUTS_ENDPOINT}/current_batch", headers={
            "Content-Type": "application/json"
        },
    )

    response.raise_for_status()

    return RootModel[list[TextToImageRequest]].model_validate_json(response.text).root


@cached(cache=TTLCache(maxsize=1, ttl=300))
def get_inputs_state() -> InputsState:
    response = requests.get(
        f"{INPUTS_ENDPOINT}/state", headers={
            "Content-Type": "application/json"
        },
    )

    response.raise_for_status()

    data = response.json()
    return InputsState(
        benchmarks_version=data["benchmarks_version"],
        delayed_weight_setting=data["delayed_weight_setting"],
        winner_percentage=data["winner_percentage"],
        active_contests={
            ContestId[contest_name.upper()]: {MetricType[metric_type.upper()]: int(weight) for metric_type, weight in metric_weights.items()}
            for contest_name, metric_weights in data["active_contests"].items()
        }
    )


@cached(cache=TTLCache(maxsize=1, ttl=300))
def get_blacklist() -> Blacklist:
    response = requests.get(
        f"{INPUTS_ENDPOINT}/blacklist", headers={
            "Content-Type": "application/json"
        },
    )

    response.raise_for_status()
    return Blacklist.model_validate(response.json())

@cached(cache=TTLCache(maxsize=1, ttl=300))
def get_duplicate_submissions() -> list[DuplicateSubmission]:
    response = requests.get(
        f"{INPUTS_ENDPOINT}/duplicate_submissions", headers={
            "Content-Type": "application/json"
        },
    )

    response.raise_for_status()
    return RootModel[list[DuplicateSubmission]].model_validate_json(response.text).root
