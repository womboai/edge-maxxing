import asyncio
import base64
import logging
import re
import json
from argparse import ArgumentParser
from pathlib import Path

from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.interface import get_substrate
from fiber.logging_utils import get_logger
from git import GitCommandError, cmd

from neuron import (
    CheckpointSubmission,
    get_config,
    find_contest,
    Contest,
    CURRENT_CONTEST,
    CONTESTS,
    ContestId,
    REVISION_LENGTH,
    make_submission,
    random_inputs,
    ModelRepositoryInfo,
    BaselineBenchmark,
    TextToImageRequest,
    MetricData,
    GenerationOutput,
    BENCHMARKS_VERSION,
)
from neuron.submission_tester import (
    generate_baseline,
    compare_checkpoints,
)

VALID_PROVIDER_REGEX = r'^[a-zA-Z0-9-.]+$'
VALID_REPO_REGEX = r'^[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+$'
VALID_REVISION_REGEX = r"^[a-f0-9]{7}$"

MODEL_DIRECTORY = Path("model")
BASELINE_MODEL_DIRECTORY = Path("baseline-model")
BASELINE_CACHE_JSON = BASELINE_MODEL_DIRECTORY / "baseline_cache.json"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = get_logger(__name__)


def add_extra_args(argument_parser: ArgumentParser):
    argument_parser.add_argument(
        "--provider",
        type=str,
        help="The git provider containing the repository",
    )

    argument_parser.add_argument(
        "--repository",
        type=str,
        help="The repository to push to",
    )

    argument_parser.add_argument(
        "--revision",
        type=str,
        help="The revision to checkout",
    )

    argument_parser.add_argument(
        "--contest",
        type=str,
        help="The contest to submit to",
    )

    argument_parser.add_argument(
        "--benchmarking.on",
        action="store_true",
        help="Turn on benchmarking.",
        default=False,
    )


def load_baseline_cache(inputs: list[TextToImageRequest]) -> BaselineBenchmark | None:
    try:
        if not BASELINE_CACHE_JSON.exists():
            return None

        with open(BASELINE_CACHE_JSON, "r") as f:
            data = json.load(f)

            benchmarks_version = data["benchmarks_version"]
            if BENCHMARKS_VERSION != benchmarks_version:
                logger.info(f"Baseline cache is outdated, regenerating baseline")
                return None

            cached_inputs = [TextToImageRequest(**input_data) for input_data in data["inputs"]]
            if cached_inputs != inputs:
                logger.info("Contest inputs have changed, regenerating baseline")
                return None

            metrics = MetricData(**data["metrics"])
            outputs = [
                GenerationOutput(
                    output=base64.b64decode(output_data["output"]),
                    generation_time=output_data["generation_time"],
                    vram_used=output_data["vram_used"],
                    watts_used=output_data["watts_used"]
                )
                for output_data in data["outputs"]
            ]
            return BaselineBenchmark(inputs=inputs, outputs=outputs, metric_data=metrics)
    except Exception as e:
        logger.error(f"Failed to load baseline cache: {e}. Clearing.")
        return None


def save_baseline_cache(baseline: BaselineBenchmark):
    with open(BASELINE_CACHE_JSON, "w") as f:
        data = {
            "benchmarks_version": BENCHMARKS_VERSION,
            "inputs": [request.model_dump(exclude_none=True) for request in baseline.inputs],
            "outputs": [
                {
                    "output": base64.b64encode(output.output).decode('utf-8'),
                    "generation_time": output.generation_time,
                    "vram_used": output.vram_used,
                    "watts_used": output.watts_used
                }
                for output in baseline.outputs
            ],
            "metrics": baseline.metric_data.model_dump(exclude_none=True),
        }
        json.dump(data, f, indent=4)


async def start_benchmarking(submission: CheckpointSubmission):
    logger.info("Generating baseline samples to compare")
    if not BASELINE_MODEL_DIRECTORY.exists():
        BASELINE_MODEL_DIRECTORY.mkdir()
    inputs = random_inputs()

    baseline = load_baseline_cache(inputs)
    if baseline is None:
        baseline = await generate_baseline(
            inputs,
            BASELINE_MODEL_DIRECTORY,
            False,
            True,
        )
        save_baseline_cache(baseline)
    else:
        logger.info("Using cached baseline")

    logger.info("Comparing submission to baseline")
    if not MODEL_DIRECTORY.exists():
        MODEL_DIRECTORY.mkdir()

    await compare_checkpoints(
        ModelRepositoryInfo(url=submission.get_repo_link(), revision=submission.revision),
        [],
        inputs,
        baseline,
        MODEL_DIRECTORY,
        False,
        True,
    )


def validate(provider: str, repository: str, revision: str, contest: Contest):
    if not re.match(VALID_REPO_REGEX, repository):
        raise ValueError(f"Invalid repository URL: {repository}")

    if not re.match(VALID_REVISION_REGEX, revision):
        raise ValueError(f"Invalid revision hash: {revision}")

    if contest not in CONTESTS:
        raise ValueError(f"Invalid contest: {contest.id.name}")

    if repository in contest.baseline_repository.url:
        raise ValueError(f"Cannot submit baseline repository: {repository}")

    if revision == contest.baseline_repository.revision:
        raise ValueError(f"Cannot submit baseline revision: {revision}")

    git = cmd.Git()
    try:
        git.ls_remote(f"https://{provider}/{repository}", revision)
    except GitCommandError as e:
        raise ValueError(f"Invalid repository or revision: {e}")


def get_latest_revision(provider: str, repository: str):
    git = cmd.Git()
    return git.ls_remote(f"https://{provider}/{repository}").split()[0][:REVISION_LENGTH]


def get_submission(config) -> CheckpointSubmission:
    provider = config["provider"]
    repository = config["repository"]
    revision = config["revision"]
    contest_name = config["contest"]
    contest: Contest | None = None

    if contest_name:
        try:
            contest = find_contest(ContestId[contest_name])
        except ValueError:
            exit(f"Unknown contest: {contest_name}")

    if not provider:
        while True:
            provider = input("Enter git provider (such as github.com or huggingface.co): ")
            if re.match(VALID_PROVIDER_REGEX, provider):
                break
            else:
                print("Invalid git provider.")

    if not repository:
        while True:
            repository = input("Enter repository URL (format: <username>/<repo>): ")
            if re.match(VALID_REPO_REGEX, repository):
                break
            else:
                print("Invalid repository URL.")

    if not revision:
        while True:
            try:
                revision = input("Enter short revision hash (leave blank to fetch latest): ") or get_latest_revision(provider, repository)
            except GitCommandError as e:
                exit(f"Failed to get latest revision: {e}")
            if re.match(VALID_REVISION_REGEX, revision):
                break
            else:
                print("Invalid revision hash. Should be 7 characters long.")

    if not contest:
        while True:
            print("\nAvailable contests:")
            for c in CONTESTS:
                print(f"\t- {c.id.name}")
            contest_id = input(f"Enter the contest (default: {CURRENT_CONTEST.id.name}): ") or CURRENT_CONTEST.id.name
            try:
                contest = find_contest(ContestId[contest_id])
                break
            except ValueError:
                print(f"Unknown contest: {contest_id}")
            except KeyError:
                print(f"Invalid contest: {contest_id}")

    try:
        validate(provider, repository, revision, contest)
    except ValueError as e:
        exit(f"Validation failed: {e}")

    return CheckpointSubmission(
        provider=provider,
        repository=repository,
        revision=revision,
        contest=contest.id,
    )


async def submit():
    config = get_config(add_extra_args)

    substrate = get_substrate(
        subtensor_network=config["subtensor.network"],
        subtensor_address=config["subtensor.chain_endpoint"]
    )

    keypair = load_hotkey_keypair(wallet_name=config["wallet.name"], hotkey_name=config["wallet.hotkey"])

    submission = get_submission(config)
    enable_benchmarking = config["benchmarking.on"]

    if enable_benchmarking or input("Benchmark submission before submitting? (y/N): ").strip().lower() in ("yes", "y"):
        await start_benchmarking(submission)

    print(
        "\nSubmission info:\n"
        f"Git Provider: {submission.provider}\n"
        f"Repository:   {submission.repository}\n"
        f"Revision:     {submission.revision}\n"
        f"Contest:      {submission.contest.name}\n"
    )
    if input("Confirm submission? (Y/n): ").strip().lower() not in ("yes", "y", ""):
        exit("Submission cancelled.")

    make_submission(
        substrate,
        config["netuid"],
        keypair,
        [submission],
    )

    logger.info(f"Submitted {submission} as the info for this miner")


def main():
    asyncio.run(submit())


if __name__ == '__main__':
    main()
