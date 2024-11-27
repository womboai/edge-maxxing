from abc import ABC, abstractmethod
from enum import Enum

class Gpu(Enum):
    NVIDIA_RTX_4090 = "NVIDIA GeForce RTX 4090"

class Device(ABC):
    @abstractmethod
    def get_vram_used(self):
        ...

    @abstractmethod
    def get_joules(self):
        ...

    @abstractmethod
    def is_compatible(self):
        ...

class CudaDevice(Device):
    _gpu: Gpu

    def __init__(self, gpu: Gpu):
        self._gpu = gpu

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

    def is_compatible(self):
        import torch

        device_name = torch.cuda.get_device_name()

        return device_name == self._gpu.value

class MpsDevice(Device):
    def get_vram_used(self):
        import torch

        return torch.mps.current_allocated_memory()

    def get_joules(self):
        return 0  # TODO

    def is_compatible(self):
        import torch

        return torch.backends.mps.is_available()

class ContestDeviceValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

        self.message = message
