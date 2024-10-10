from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from io import BytesIO
from typing import TypeVar, Callable

from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

RequestT = TypeVar("RequestT", bound=BaseModel)
ResponseT = TypeVar("ResponseT")


class ModelRepositoryInfo(BaseModel):
    url: str
    revision: str


class ContestId(Enum):
    SDXL_APPLE_SILICON = 0
    SDXL_NEWDREAM_NVIDIA_4090 = 1
    FLUX_NVIDIA_4090 = 2


class OutputComparator(ABC):
    @abstractmethod
    def compare(self, baseline: bytes, optimized: bytes) -> float:
        pass

    def __wrapped_compare(self, baseline: bytes, optimized: bytes):
        return self.compare(baseline, optimized)

    __call__ = __wrapped_compare


class ImageOutputComparator(OutputComparator):
    def __init__(self, device: str):
        self.device = device
        self.clip = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    def compare(self, baseline: bytes, optimized: bytes):
        from torch import manual_seed
        from torch.nn.functional import cosine_similarity

        from PIL import Image

        from skimage import metrics
        import cv2

        import numpy

        manual_seed(0)

        def load_image(data: bytes):
            with BytesIO(data) as fp:
                return numpy.array(Image.open(fp).convert("RGB"))

        def clip_embeddings(image: numpy.ndarray):
            processed_input = self.processor(images=image, return_tensors="pt").to(self.device)

            return self.clip(**processed_input).image_embeds.to(self.device)

        baseline_array = load_image(baseline)
        optimized_array = load_image(optimized)

        grayscale_baseline = cv2.cvtColor(baseline_array, cv2.COLOR_RGB2GRAY)
        grayscale_optimized = cv2.cvtColor(optimized_array, cv2.COLOR_RGB2GRAY)

        structural_similarity = metrics.structural_similarity(grayscale_baseline, grayscale_optimized, full=True)[0]

        del grayscale_baseline
        del grayscale_optimized

        baseline_embeddings = clip_embeddings(baseline_array)
        optimized_embeddings = clip_embeddings(optimized_array)

        clip_similarity = cosine_similarity(baseline_embeddings, optimized_embeddings)[0].item()

        return clip_similarity * 0.35 + structural_similarity * 0.65


class Contest(ABC):
    id: ContestId
    output_comparator: Callable[[], OutputComparator]
    baseline_repository: ModelRepositoryInfo

    def __init__(
        self,
        contest_id: ContestId,
        output_comparator: Callable[[], OutputComparator],
        baseline_repository: ModelRepositoryInfo,
    ):
        self.id = contest_id
        self.output_comparator = output_comparator
        self.baseline_repository = baseline_repository

    @abstractmethod
    def get_vram_used(self) -> int:
        ...

    @abstractmethod
    def get_joules(self) -> float:
        ...

    @abstractmethod
    def validate(self) -> None:
        ...


class CudaContest(Contest):
    def __init__(
        self,
        contest_id: ContestId,
        baseline_repository: ModelRepositoryInfo,
        expected_device_name: str,
    ):
        super().__init__(contest_id, partial(ImageOutputComparator, "cuda"), baseline_repository)

        self.expected_device_name = expected_device_name

    def get_vram_used(self):
        import pynvml
        import torch

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        pynvml.nvmlShutdown()
        return vram

    def get_joules(self):
        import pynvml
        import torch

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        pynvml.nvmlShutdown()
        return mj / 1000.0  # convert mJ to J

    def validate(self):
        import torch

        device_name = torch.cuda.get_device_name()

        if device_name != self.expected_device_name:
            raise ContestDeviceValidationError(
                f"Incompatible device {device_name} when {self.expected_device_name} is required.",
            )


class AppleSiliconContest(Contest):
    def __init__(
        self,
        contest_id: ContestId,
        baseline_repository: ModelRepositoryInfo,
    ):
        super().__init__(contest_id, partial(ImageOutputComparator, "mps"), baseline_repository)

    def get_vram_used(self):
        import torch

        return torch.mps.current_allocated_memory()

    def get_joules(self):
        return 0  # TODO

    def validate(self):
        import torch

        if not torch.backends.mps.is_available():
            raise ContestDeviceValidationError("MPS is not available but is required.")


class ContestDeviceValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

        self.message = message


CONTESTS = [
    CudaContest(
        ContestId.SDXL_NEWDREAM_NVIDIA_4090,
        ModelRepositoryInfo(url="https://github.com/womboai/sdxl-newdream-20-inference", revision="3e5710d"),
        "NVIDIA GeForce RTX 4090",
    ),
]


def find_contest(contest_id: ContestId):
    for c in CONTESTS:
        if c.id != contest_id:
            continue

        return c

    raise RuntimeError(f"Unknown contest ID requested {contest_id}")


CURRENT_CONTEST: CudaContest = find_contest(ContestId.SDXL_NEWDREAM_NVIDIA_4090)
