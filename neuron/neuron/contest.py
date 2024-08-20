from abc import ABC, abstractmethod
from enum import Enum

import torch
from diffusers import StableDiffusionXLPipeline

from .coreml_pipeline import CoreMLStableDiffusionXLPipeline


class ContestId(Enum):
    APPLE_SILICON = 0
    NVIDIA_4090 = 1


class Contest(ABC):
    id: ContestId
    baseline_average: float
    baseline_repository: str
    device_name: str | None

    def __init__(self, contest_id: ContestId, baseline_average: float, baseline_repository: str):
        self.id = contest_id
        self.baseline_average = baseline_average
        self.baseline_repository = baseline_repository

    def load_baseline(self):
        return self.load()

    @abstractmethod
    def load(self, repository: str | None = None):
        ...

    @abstractmethod
    def validate(self):
        ...

    @abstractmethod
    def empty_cache(self):
        ...


class CudaContest(Contest):
    def __init__(self, contest_id: ContestId, baseline_average: float, baseline_repository: str, expected_device_name: str):
        super().__init__(contest_id, baseline_average, baseline_repository)

        self.expected_device_name = expected_device_name

    def load(self, repository: str | None = None):
        return StableDiffusionXLPipeline.from_pretrained(
            repository or self.baseline_repository,
            torch_dtype=torch.float16,
            use_safetensors=True if repository else None,
        ).to("cuda")

    def validate(self):
        device_name = torch.cuda.get_device_name("cuda")

        if device_name != self.expected_device_name:
            raise ContestDeviceValidationError(
                f"Incompatible device {device_name} when {self.expected_device_name} is required.",
            )

    def empty_cache(self):
        torch.cuda.empty_cache()


class AppleSiliconContest(Contest):
    def load(self, repository: str | None = None):
        return CoreMLStableDiffusionXLPipeline.from_pretrained(repository or self.baseline_repository).to("mps")

    def validate(self):
        if not torch.backends.mps.is_available():
            raise ContestDeviceValidationError("mps is not available but is required.")

    def empty_cache(self):
        torch.mps.empty_cache()


class ContestDeviceValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

        self.message = message


CONTESTS = [
    AppleSiliconContest(ContestId.APPLE_SILICON, 2.5, "wombo/coreml-stable-diffusion-xl-base-1.0"),
    CudaContest(ContestId.NVIDIA_4090, 2.58, "stablediffusionapi/newdream-sdxl-20", "NVIDIA GeForce RTX 4090"),
]


def find_contest(contest_id: ContestId):
    for c in CONTESTS:
        if c.id != contest_id:
            continue

        return c

    raise RuntimeError(f"Unknown contest ID requested {contest_id}")


CURRENT_CONTEST = find_contest(ContestId.NVIDIA_4090)
