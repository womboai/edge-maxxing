import os
from zoneinfo import ZoneInfo

import requests
from pydantic import RootModel

from pipelines import TextToImageRequest

TIMEZONE = ZoneInfo("America/Los_Angeles")
INPUTS_ENDPOINT = os.getenv("INPUTS_ENDPOINT", "https://edge-inputs.api.wombo.ai")


def random_inputs() -> list[TextToImageRequest]:
    response = requests.get(
        f"{INPUTS_ENDPOINT}/current_batch", headers={
            "Content-Type": "application/json"
        },
    )

    response.raise_for_status()

    return RootModel[list[TextToImageRequest]].model_validate_json(response.text).root
