from zoneinfo import ZoneInfo

import requests
from pydantic import RootModel

from pipelines import TextToImageRequest

INFERENCE_SOCKET_TIMEOUT = 240
TIMEZONE = ZoneInfo("US/Pacific")


def random_inputs() -> list[TextToImageRequest]:
    response = requests.get(
        "https://edge-inputs.wombo.ai/current_batch", headers={
            "Content-Type": "application/json"
        },
    )

    response.raise_for_status()

    return RootModel[list[TextToImageRequest]].model_validate_json(response.text).root
