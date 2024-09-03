from abc import ABC, abstractmethod
from enum import Enum
from io import BytesIO
from typing import Generic, Callable, TypeVar

import torch
from PIL import Image
from pydantic import BaseModel

from pipelines.pipelines.models import TextToImageRequest

RequestT = TypeVar("RequestT", bound=BaseModel)
ResponseT = TypeVar("ResponseT")


def image_response_deserializer(data: bytes):
    with BytesIO(data) as fp:
        return Image.open(fp)


class ContestId(Enum):
    APPLE_SILICON = 0
    NVIDIA_4090 = 1


class Contest(Generic[ResponseT], ABC):
    id: ContestId
    baseline_image: str
    device_name: str | None
    response_deserializer: Callable[[bytes], ResponseT]

    def __init__(
        self,
        contest_id: ContestId,
        baseline_repository: str,
        response_deserializer: Callable[[bytes], ResponseT],
    ):
        self.id = contest_id
        self.baseline_image = baseline_repository
        self.response_deserializer = response_deserializer

    @abstractmethod
    def validate(self):
        ...


class CudaContest(Contest[TextToImageRequest, Image.Image]):
    def __init__(
        self,
        contest_id: ContestId,
        baseline_repository: str,
        expected_device_name: str,
        response_deserializer: Callable[[bytes], ResponseT],
    ):
        super().__init__(contest_id, baseline_repository, response_deserializer)

        self.expected_device_name = expected_device_name

    def validate(self):
        device_name = torch.cuda.get_device_name("cuda")

        if device_name != self.expected_device_name:
            raise ContestDeviceValidationError(
                f"Incompatible device {device_name} when {self.expected_device_name} is required.",
            )


class AppleSiliconContest(Contest[TextToImageRequest, Image.Image]):
    def validate(self):
        if not torch.backends.mps.is_available():
            raise ContestDeviceValidationError("MPS is not available but is required.")


class ContestDeviceValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

        self.message = message


CONTESTS = [
    CudaContest(
        ContestId.NVIDIA_4090,
        "womboashley/edge-maxxing-miner-models:newdream",
        "NVIDIA GeForce RTX 4090",
        image_response_deserializer,
    ),
]


def find_contest(contest_id: ContestId):
    for c in CONTESTS:
        if c.id != contest_id:
            continue

        return c

    raise RuntimeError(f"Unknown contest ID requested {contest_id}")


CURRENT_CONTEST: CudaContest = find_contest(ContestId.NVIDIA_4090)
