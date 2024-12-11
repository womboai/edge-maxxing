import psutil
from pydantic import BaseModel


class SystemInfo(BaseModel):
    cpu: str
    min_frequency_mhz: float
    max_frequency_mhz: float
    current_frequency_mhz: float
    physical_cores: int
    total_cores: int
    ram: int
    gpu: str


def get_system_info() -> SystemInfo:
    with open("/proc/cpuinfo", "r") as f:
        for line in f:
            if "model name" in line:
                cpu = line.strip().split(":")[1].strip()
                break

    cpu_frequency = psutil.cpu_freq()
    min_frequency = cpu_frequency.min
    max_frequency = cpu_frequency.max
    current_frequency = cpu_frequency.current

    physical_cores = psutil.cpu_count(logical=False)
    total_cores = psutil.cpu_count(logical=True)

    ram = psutil.virtual_memory().total

    import torch
    gpu = torch.cuda.get_device_name(0)

    return SystemInfo(
        cpu=cpu,
        min_frequency_mhz=min_frequency,
        max_frequency_mhz=max_frequency,
        current_frequency_mhz=current_frequency,
        physical_cores=physical_cores,
        total_cores=total_cores,
        ram=ram,
        gpu=gpu,
    )
