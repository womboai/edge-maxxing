import re
from argparse import ArgumentParser

from git import GitCommandError, cmd

from neuron import (
    bt,
    CheckpointSubmission,
    get_config,
    find_contest,
    Contest,
    CURRENT_CONTEST,
    CONTESTS,
    ContestId,
    REVISION_LENGTH,
)

from neuron.submissions import make_submission

VALID_PROVIDER_REGEX = r'^[a-zA-Z0-9-.]+$'
VALID_REPO_REGEX = r'^[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+$'
VALID_REVISION_REGEX = r"^[a-f0-9]{7}$"


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


def validate(provider: str, repository: str, revision: str, contest: Contest):
    if not re.match(VALID_REPO_REGEX, repository):
        raise ValueError(f"Invalid repository URL: {repository}")

    if not re.match(VALID_REVISION_REGEX, revision):
        raise ValueError(f"Invalid revision hash: {revision}")

    if contest not in CONTESTS:
        raise ValueError(f"Invalid contest: {contest.id.name}")

    if contest.baseline_repository == repository:
        raise ValueError(f"Cannot submit baseline repository: {repository}")
    if contest.baseline_revision == revision:
        raise ValueError(f"Cannot submit baseline revision: {revision}")

    git = cmd.Git()
    try:
        git.ls_remote(f"https://{provider}/{repository}", revision)
    except GitCommandError as e:
        raise ValueError(f"Invalid repository or revision: {e}")


def get_latest_revision(provider: str, repository: str):
    git = cmd.Git()
    return git.ls_remote(f"https://{provider}/{repository}").split()[0][:REVISION_LENGTH]


def main():
    config = get_config(add_extra_args)

    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    wallet = bt.wallet(config=config)

    provider = config.provider
    repository = config.repository
    revision = config.revision
    contest: Contest | None = None

    if config.contest:
        try:
            contest = find_contest(ContestId[config.contest])
        except ValueError:
            exit(f"Unknown contest: {config.contest}")

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

    checkpoint_info = CheckpointSubmission(
        provider=provider,
        repository=repository,
        revision=revision,
        contest=contest.id,
    )

    print(
        "\nSubmission info:\n"
        f"Git Provider: {checkpoint_info.provider}\n"
        f"Repository:   {checkpoint_info.repository}\n"
        f"Revision:     {checkpoint_info.revision}\n"
        f"Contest:      {checkpoint_info.contest.name}\n"
    )
    if input("Confirm submission? (Y/n): ").strip().lower() not in ("yes", "y", ""):
        exit("Submission cancelled.")

    make_submission(
        subtensor,
        metagraph,
        wallet,
        [checkpoint_info],
    )

    bt.logging.info(f"Submitted {checkpoint_info} as the info for this miner")


if __name__ == '__main__':
    main()
