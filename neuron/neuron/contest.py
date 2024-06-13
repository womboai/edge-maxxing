from enum import Enum
from typing import Callable, NoReturn

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

from .coreml_pipeline import CoreMLStableDiffusionXLPipeline


class ContestId(Enum):
    APPLE_SILICON = 0
    NVIDIA_4090 = 1


class Contest:
    id: ContestId
    baseline_repository: str
    device_name: str | None
    load: Callable[[str], DiffusionPipeline]
    validate: Callable[[], None]
    empty_cache: Callable[[], None]

    def __init__(
        self,
        contest_id: ContestId,
        baseline_repository: str,
        loader: Callable[[str], DiffusionPipeline],
        validate: Callable[[], None],
        empty_cache: Callable[[], None],
    ):
        self.id = contest_id
        self.baseline_repository = baseline_repository
        self.load = loader
        self.validate = validate
        self.empty_cache = empty_cache


class ContestDeviceValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

        self.message = message


def check_mps_availability():
    if not torch.mps.is_available():
        raise ContestDeviceValidationError("mps is not available but is required.")


def check_cuda_device_name(expected_name: str) -> None | NoReturn:
    device_name = torch.cuda.get_device_name("cuda")

    if device_name != expected_name:
        raise ContestDeviceValidationError(f"Incompatible device {device_name} when {expected_name} is required.")


CONTESTS = [
    Contest(
        ContestId.APPLE_SILICON,
        "wombo/coreml-stable-diffusion-xl-base-1.0",
        lambda repository: CoreMLStableDiffusionXLPipeline.from_pretrained(repository).to("mps"),
        lambda: check_mps_availability(),
        lambda: torch.mps.empty_cache(),
    ),
    Contest(
        ContestId.NVIDIA_4090,
        "stablediffusionapi/newdream-sdxl-20",
        lambda repository: StableDiffusionXLPipeline.from_pretrained(repository, torch_dtype=torch.float16).to("cuda"),
        lambda: check_cuda_device_name("NVIDIA GeForce RTX 4090"),
        lambda: torch.cuda.empty_cache(),
    ),
]


def find_contest(contest_id: ContestId):
    for c in CONTESTS:
        if c.id != contest_id:
            continue

        return c

    raise RuntimeError(f"Unknown contest ID requested {contest_id}")


CURRENT_CONTEST = find_contest(ContestId.NVIDIA_4090)
