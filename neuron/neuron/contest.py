import shutil
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import constants

from .coreml_pipeline import CoreMLStableDiffusionXLPipeline

MODEL_CACHE_DIR = Path("model-cache")


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

    def delete_model_cache(self):
        if MODEL_CACHE_DIR.exists():
            shutil.rmtree(MODEL_CACHE_DIR)

    def get_model_size(self):
        return sum(file.stat().st_size for file in MODEL_CACHE_DIR.rglob("*"))

    @abstractmethod
    def load(self, repository: str | None = None):
        ...

    @abstractmethod
    def get_baseline_size(self):
        ...

    @abstractmethod
    def get_vram_used(self, device: torch.device):
        ...

    @abstractmethod
    def validate(self):
        ...

    @abstractmethod
    def empty_cache(self):
        ...


class CudaContest(Contest):
    def __init__(self, contest_id: ContestId, baseline_average: float, baseline_repository: str,
                 expected_device_name: str):
        super().__init__(contest_id, baseline_average, baseline_repository)

        self.expected_device_name = expected_device_name

    def load(self, repository: str | None = None):
        return StableDiffusionXLPipeline.from_pretrained(
            repository or self.baseline_repository,
            torch_dtype=torch.float16,
            use_safetensors=True if repository else None,
            cache_dir=MODEL_CACHE_DIR if repository else None,
        ).to("cuda")

    def get_baseline_size(self):
        baseline_dir = Path(constants.HF_HUB_CACHE) / f"models--{self.baseline_repository.replace('/', '--')}"
        return sum(file.stat().st_size for file in baseline_dir.rglob("*"))

    def get_vram_used(self, device: torch.device):
        return torch.cuda.memory_allocated(device)

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

    def get_baseline_size(self):
        return 0 # TODO

    def get_vram_used(self, device: torch.device):
        return torch.mps.current_allocated_memory()

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
