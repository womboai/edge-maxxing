import os

import requests
from pydantic import RootModel

from pipelines import TextToImageRequest

INPUTS_ENDPOINT = os.getenv("INPUTS_ENDPOINT", "https://edge-inputs.api.wombo.ai")

def random_inputs() -> list[TextToImageRequest]:
    response = requests.get(
        f"{INPUTS_ENDPOINT}/current_batch", headers={
            "Content-Type": "application/json"
        },
    )

    response.raise_for_status()

    return RootModel[list[TextToImageRequest]].model_validate_json(response.text).root

def blacklisted_keys() -> dict:
    response = requests.get(
        f"{INPUTS_ENDPOINT}/blacklist", headers={
            "Content-Type": "application/json"
        },
    )

    response.raise_for_status()
    return response.json()

def is_blacklisted(blacklist: dict, hotkey: str, coldkey: str):
    return hotkey in blacklist["hotkeys"] or coldkey in blacklist["coldkeys"]
