import os
from multiprocessing.connection import Client
from os.path import abspath
from pathlib import Path
from subprocess import Popen
from time import sleep, perf_counter

from fiber.logging_utils import get_logger

from neuron import (
    CheckpointSubmission,
    find_contest,
    Contest,
    GenerationOutput,
    get_config,
    generate_random_prompt,
    VRamMonitor,
    INFERENCE_SOCKET_TIMEOUT,
    BENCHMARK_SAMPLE_COUNT,
    setup_sandbox,
    random_seed,
)
from pipelines import TextToImageRequest

MODEL_DIRECTORY = Path("model")

logger = get_logger(__name__)


def wait_for_socket(socket_path: str, process: Popen):
    for _ in range(INFERENCE_SOCKET_TIMEOUT):
        if os.path.exists(socket_path):
            break

        sleep(1)

        if process.returncode:
            raise RuntimeError(f"Model crashed with exit code {process.returncode}")
    else:
        raise RuntimeError(f"Socket file '{socket_path}' not found after {INFERENCE_SOCKET_TIMEOUT} seconds. Your pipeline is taking too long to load. Please optimize and avoid precompiling if possible.")


def test(contest: Contest, client: Client):
    outputs: list[GenerationOutput] = []
    for i in range(BENCHMARK_SAMPLE_COUNT):
        output = benchmark(contest, client)

        logger.info(
            f"Sample {i} Generated\n"
            f"Generation Time: {output.generation_time}s\n"
            f"VRAM Usage: {output.vram_used}b\n"
            f"Power Usage: {output.watts_used}W"
        )

        outputs.append(output)

    size = sum(file.stat().st_size for file in MODEL_DIRECTORY.rglob("*"))
    average_time = sum(output.generation_time for output in outputs) / len(outputs)
    vram_used = max(output.vram_used for output in outputs)
    watts_used = max(output.watts_used for output in outputs)

    logger.info(
        f"\n\nTested {BENCHMARK_SAMPLE_COUNT} Samples\n"
        f"Average Generation Time: {average_time}s\n"
        f"Model Size: {size}b\n"
        f"Max VRAM Usage: {vram_used}b\n"
        f"Max Power Usage: {watts_used}W"
    )


def benchmark(contest: Contest, client: Client):
    start_joules = contest.get_joules()
    vram_monitor = VRamMonitor(contest)
    start = perf_counter()

    prompt = generate_random_prompt()
    seed = random_seed()

    request = TextToImageRequest(
        prompt=prompt,
        seed=seed,
    )

    data = request.model_dump_json().encode("utf-8")
    client.send_bytes(data)
    output = client.recv_bytes()

    generation_time = perf_counter() - start
    joules_used = contest.get_joules() - start_joules
    watts_used = joules_used / generation_time
    vram_used = vram_monitor.complete()

    return GenerationOutput(
        prompt=prompt,
        seed=seed,
        output=output,
        generation_time=generation_time,
        vram_used=vram_used,
        watts_used=watts_used,
    )


def start_benchmarking(submission: CheckpointSubmission):
    contest = find_contest(submission.contest)
    contest.validate()
    logger.info(f"Benchmarking '{submission.get_repo_link()}' with revision '{submission.revision}'")

    if not MODEL_DIRECTORY.exists():
        MODEL_DIRECTORY.mkdir()

    setup_sandbox([], MODEL_DIRECTORY, True, submission.get_repo_link(), submission.revision)

    socket_path = abspath(MODEL_DIRECTORY / "inferences.sock")

    with Popen([abspath(MODEL_DIRECTORY / ".venv" / "bin" / "start_inference")], cwd=MODEL_DIRECTORY) as process:
        logger.info(f"Inference process starting")
        wait_for_socket(socket_path, process)

        logger.info("Connecting to socket")
        with Client(socket_path) as client:
            test(contest, client)


if __name__ == '__main__':
    from .submit import add_extra_args, get_submission
    config = get_config(add_extra_args)
    submission = get_submission(config)
    start_benchmarking(submission)
