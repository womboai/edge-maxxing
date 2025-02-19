import mmap
import os
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
from cachetools import cached, TTLCache
from pydantic import BaseModel


class SystemInfo(BaseModel):
    cpu: str
    min_frequency_mhz: float
    max_frequency_mhz: float
    current_frequency_mhz: float
    physical_cores: int
    total_cores: int
    ram: int
    ram_speed: float
    gpu: str
    total_storage: int
    used_storage: int
    free_storage: int
    ssd_read_speed: float
    ssd_write_speed: float


@cached(cache=TTLCache(maxsize=1, ttl=300))
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

    disk_usage = shutil.disk_usage(os.getcwd())
    read_speed, write_speed = calculate_ssd_speed()

    return SystemInfo(
        cpu=cpu,
        min_frequency_mhz=min_frequency,
        max_frequency_mhz=max_frequency,
        current_frequency_mhz=current_frequency,
        physical_cores=physical_cores,
        total_cores=total_cores,
        ram=ram,
        ram_speed=calculate_ram_speed(),
        gpu=gpu,
        total_storage=disk_usage.total,
        used_storage=disk_usage.used,
        free_storage=disk_usage.free,
        ssd_read_speed=read_speed,
        ssd_write_speed=write_speed,
    )


@cached(cache=TTLCache(maxsize=1, ttl=86400))
def calculate_ram_speed() -> float:
    size = 1024 * 1024 * 100
    a = np.random.random(size // 8)
    b = np.random.random(size // 8)
    np.random.random(size // 8)
    start = perf_counter()
    c = a + b
    c.sum()
    end = perf_counter()
    return (size * 3) / (end - start)


@cached(cache=TTLCache(maxsize=1, ttl=86400))
def calculate_ssd_speed() -> tuple[float, float]:
    test_file = Path("speedtest.tmp")
    chunk_size = 1024 * 1024
    total_size = chunk_size * 10

    try:
        start = perf_counter()
        with open(test_file, "wb") as f:
            f.write(b"\0" * total_size)
            f.flush()
            os.fsync(f.fileno())
        end = perf_counter()
        write_speed = total_size / (end - start)

        with open(test_file, "rb") as f:
            with mmap.mmap(f.fileno(), total_size, access=mmap.ACCESS_READ) as mm:
                start = perf_counter()
                mm.read(total_size)
                end = perf_counter()
        read_speed = total_size / (end - start)

        return read_speed, write_speed
    finally:
        test_file.unlink()
