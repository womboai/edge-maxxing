from enum import Enum
from os.path import isdir, join
from typing import Callable

from coremltools import ComputeUnit
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from huggingface_hub import snapshot_download
from python_coreml_stable_diffusion.pipeline import get_coreml_pipe

from .pipeline import StableDiffusionXLMinimalPipeline

CheckpointLoader = Callable[[str, str], DiffusionPipeline]


class ContestId(Enum):
    APPLE_SILICON = 0
    NVIDIA_4090 = 1


class Contest:
    id: ContestId
    baseline_repository: str
    device: str
    loader: CheckpointLoader

    def __init__(self, contest_id: ContestId, baseline_repository: str, device: str, loader: CheckpointLoader):
        self.id = contest_id
        self.baseline_repository = baseline_repository
        self.device = device
        self.loader = loader


CONTESTS = [
    Contest(
        ContestId.APPLE_SILICON,
        "wombo/coreml-stable-diffusion-xl-base-1.0",
        "mps",
        lambda repository, device: apple_silicon_from_pretrained(repository, device),
    ),
    Contest(
        ContestId.NVIDIA_4090,
        "stabilityai/stable-diffusion-xl-base-1.0",
        "cuda",
        lambda repository, device: StableDiffusionXLPipeline.from_pretrained(repository).to(device),
    ),
]


def find_contest(contest_id: ContestId):
    for c in CONTESTS:
        if c.id != contest_id:
            continue

        return c

    raise RuntimeError(f"Unknown contest ID requested {contest_id}")


CURRENT_CONTEST = find_contest(ContestId.APPLE_SILICON)


def apple_silicon_from_pretrained(name: str, device: str):
    base_pipeline = StableDiffusionXLMinimalPipeline.from_pretrained(name).to(device)

    if isdir(name):
        directory = name
    else:
        directory = snapshot_download(name)

    coreml_dir = join(directory, "mlpackages")

    pipeline = get_coreml_pipe(
        pytorch_pipe=base_pipeline,
        mlpackages_dir=coreml_dir,
        model_version="xl",
        compute_unit=ComputeUnit.CPU_AND_GPU.name,
        delete_original_pipe=False,
        force_zeros_for_empty_prompt=base_pipeline.force_zeros_for_empty_prompt,
    )

    return pipeline
