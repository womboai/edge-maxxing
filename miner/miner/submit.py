import re
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from base.config import get_config
from base.contest import Contest, find_contest, CONTESTS, ContestId, RepositoryInfo
from base.inputs_api import get_inputs_state
from base.submissions import CheckpointSubmission, make_submission
from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.interface import get_substrate
from fiber.logging_utils import get_logger
from git import GitCommandError, cmd
from substrateinterface import Keypair
from testing.benchmarker import Benchmarker

VALID_REPO_REGEX = r"^https:\/\/[a-zA-Z0-9.-]+\/[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+$"
VALID_REVISION_REGEX = r"^[a-f0-9]{7}$"

logger = get_logger(__name__)


def start_benchmarking(contest: Contest, keypair: Keypair, submission: CheckpointSubmission):
    if not contest.device.is_compatible():
        logger.warning("Benchmarking on an incompatible device. Results will not be accurate.")

    with TemporaryDirectory() as temp_dir:
        benchmarker = Benchmarker(
            sandbox_directory=Path(temp_dir),
            sandbox_args=[]
        )

        repository_info = RepositoryInfo(url=submission.repository, revision=submission.revision)

        benchmarker.benchmark_submissions(
            contest=contest,
            submissions={keypair.ss58_address: repository_info},
        )

        print("\nBenchmarking results:")
        print(benchmarker.benchmarks[keypair.ss58_address])


def add_extra_args(argument_parser: ArgumentParser):
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


def validate(repository: str, revision: str, contest: Contest):
    if not re.match(VALID_REPO_REGEX, repository):
        raise ValueError(f"Invalid repository URL: {repository}")

    if not re.match(VALID_REVISION_REGEX, revision):
        raise ValueError(f"Invalid revision hash: {revision}")

    if contest not in CONTESTS:
        raise ValueError(f"Invalid contest: {contest.id.name}")

    if repository in contest.baseline_repository.url:
        raise ValueError(f"Cannot submit baseline repository: {repository}")

    git = cmd.Git()
    try:
        git.ls_remote(repository, revision)
    except GitCommandError as e:
        raise ValueError(f"Invalid repository or revision: {e}")


def get_latest_revision(repository: str):
    git = cmd.Git()
    return git.ls_remote(repository).split()[0]


def get_submission(config) -> CheckpointSubmission:
    repository = config["repository"]
    revision = config["revision"]
    contest_name = config["contest"]
    contest: Contest | None = find_contest(ContestId[contest_name]) if contest_name else None

    if not repository:
        while True:
            repository = input("Enter repository URL (format: https://<git-provider>/<username>/<repo>): ")
            if re.match(VALID_REPO_REGEX, repository):
                break
            else:
                print("Invalid repository URL.")

    if not revision:
        while True:
            try:
                revision = input("Enter short revision hash (leave blank to fetch latest): ") or get_latest_revision(repository)
            except GitCommandError as e:
                exit(f"Failed to get latest revision: {e}")
            if re.match(VALID_REVISION_REGEX, revision):
                break
            else:
                print("Invalid revision hash. Should be 7 characters long.")

    if not contest:
        default_contest_id = next(iter(get_inputs_state().active_contests.keys()))
        while True:
            print("\nAvailable contests:")
            for c in CONTESTS:
                print(f"\t- {c.id.name}")
            contest_id = input(f"Enter the contest (default: {default_contest_id.name}): ") or default_contest_id.name
            try:
                contest = find_contest(ContestId[contest_id])
                break
            except ValueError:
                print(f"Unknown contest: {contest_id}")
            except KeyError:
                print(f"Invalid contest: {contest_id}")

    try:
        validate(repository, revision, contest)
    except ValueError as e:
        exit(f"Validation failed: {e}")

    return CheckpointSubmission(
        repository=repository,
        revision=revision,
        contest_id=contest.id,
    )


def submit():
    config = get_config(add_extra_args)

    substrate = get_substrate(
        subtensor_network=config["subtensor.network"],
        subtensor_address=config["subtensor.chain_endpoint"],
    )

    keypair = load_hotkey_keypair(
        wallet_name=config["wallet.name"],
        hotkey_name=config["wallet.hotkey"],
    )

    submission = get_submission(config)
    enable_benchmarking = config["benchmarking.on"]

    if enable_benchmarking or input("Benchmark submission before submitting? (y/N): ").strip().lower() in ("yes", "y"):
        try:
            start_benchmarking(find_contest(submission.contest_id), keypair, submission)
        except Exception as e:
            logger.critical(f"Benchmarking failed, submission cancelled", exc_info=e)
            exit(1)

    print(
        "\nSubmission info:\n"
        f"Repository:   {submission.repository}\n"
        f"Revision:     {submission.revision}\n"
        f"Contest:      {submission.contest_id.name}\n"
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
    submit()


if __name__ == '__main__':
    main()
