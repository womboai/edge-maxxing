from abc import ABC, abstractmethod
from enum import Enum
from io import BytesIO
from typing import TypeVar

from PIL import Image
from pydantic import BaseModel

from pipelines.models import TextToImageRequest

RequestT = TypeVar("RequestT", bound=BaseModel)
ResponseT = TypeVar("ResponseT")


class ContestId(Enum):
    APPLE_SILICON = 0
    NVIDIA_4090 = 1


class Contest(ABC):
    id: ContestId
    baseline_repository: str
    baseline_revision: str
    device_name: str | None

    def __init__(
        self,
        contest_id: ContestId,
        baseline_repository: str,
        baseline_revision: str,
    ):
        self.id = contest_id
        self.baseline_repository = baseline_repository
        self.baseline_revision = baseline_revision

    @abstractmethod
    def get_vram_used(self) -> int:
        ...

    @abstractmethod
    def get_joules(self) -> float:
        ...

    @abstractmethod
    def validate(self) -> None:
        ...

    @abstractmethod
    def compare_outputs(self, baseline: bytes, optimized: bytes) -> float:
        ...


@abstractmethod
class ImageContestMixIn(Contest, ABC):
    def compare_outputs(self, baseline: bytes, optimized: bytes) -> float:
        from torch import Tensor
        from torch.nn import Sequential
        from torch.nn.functional import cosine_similarity

        from torchvision.models import resnet50, ResNet50_Weights
        from torchvision.nn.functional import pil_to_tensor

        resnet_embed = Sequential(*list(resnet50().eval().children())[:-1])

        transform = ResNet50_Weights.DEFAULT.transforms()

        def load_image(data: bytes):
            with BytesIO(data) as fp:
                return pil_to_tensor(Image.open(fp))

        def resnet_embed_image(tensor: Tensor):
            return resnet_embed(transform(tensor).unsqueeze(0))

        baseline_tensor = load_image(baseline)
        optimized_tensor = load_image(optimized)

        resnet_similarity = cosine_similarity(
            resnet_embed_image(baseline_tensor),
            resnet_embed_image(optimized_tensor),
        ).flatten().item()

        flat_similarity = (cosine_similarity(
            (baseline_tensor / 255.0).flatten(),
            (optimized_tensor / 255.0).flatten(),
            dim=0,
        )).item()

        return (resnet_similarity ** 1024 * 0.9) + (flat_similarity ** 16 * 0.1)


class CudaContest(ImageContestMixIn, Contest):
    def __init__(
        self,
        contest_id: ContestId,
        baseline_repository: str,
        baseline_revision: str,
        expected_device_name: str,
    ):
        super().__init__(contest_id, baseline_repository, baseline_revision)

        self.expected_device_name = expected_device_name

    def get_vram_used(self):
        import torch

        return torch.cuda.memory_allocated()

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

        device_name = torch.cuda.get_device_name("cuda")

        if device_name != self.expected_device_name:
            raise ContestDeviceValidationError(
                f"Incompatible device {device_name} when {self.expected_device_name} is required.",
            )


class AppleSiliconContest(ImageContestMixIn, Contest):
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
        ContestId.NVIDIA_4090,
        "https://github.com/womboai/sdxl-newdream-20-inference",
        "d79d4af18f9f5dac7fcb310bb8009815ecc63d57",
        "NVIDIA GeForce RTX 4090",
    ),
]


def find_contest(contest_id: ContestId):
    for c in CONTESTS:
        if c.id != contest_id:
            continue

        return c

    raise RuntimeError(f"Unknown contest ID requested {contest_id}")


CURRENT_CONTEST: CudaContest = find_contest(ContestId.NVIDIA_4090)
