from enum import Enum
from typing import Callable

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

from .coreml_pipeline import CoreMLStableDiffusionXLPipeline

CheckpointLoader = Callable[[str, str], DiffusionPipeline]


class ContestId(Enum):
    APPLE_SILICON = 0
    NVIDIA_4090 = 1


class Contest:
    id: ContestId
    baseline_repository: str
    device: str
    device_name: str | None
    loader: CheckpointLoader

    def __init__(
        self,
        contest_id: ContestId,
        baseline_repository: str,
        device: str,
        device_name: str | None,
        loader: CheckpointLoader,
    ):
        self.id = contest_id
        self.baseline_repository = baseline_repository
        self.device = device
        self.device_name = device_name
        self.loader = loader


CONTESTS = [
    Contest(
        ContestId.APPLE_SILICON,
        "wombo/coreml-stable-diffusion-xl-base-1.0",
        "mps",
        None,
        lambda repository, device: CoreMLStableDiffusionXLPipeline.from_pretrained(repository).to(device),
    ),
    Contest(
        ContestId.NVIDIA_4090,
        "stablediffusionapi/newdream-sdxl-20",
        "cuda",
        "NVIDIA GeForce RTX 4090",
        lambda repository, device: StableDiffusionXLPipeline.from_pretrained(
            repository,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device),
    ),
]


def find_contest(contest_id: ContestId):
    for c in CONTESTS:
        if c.id != contest_id:
            continue

        return c

    raise RuntimeError(f"Unknown contest ID requested {contest_id}")


CURRENT_CONTEST = find_contest(ContestId.NVIDIA_4090)
