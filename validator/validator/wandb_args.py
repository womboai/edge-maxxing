from argparse import ArgumentParser


def add_wandb_args(parser: ArgumentParser):
    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=False,
    )

    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )

    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="Wandb entity to log to.",
        default="w-ai-wombo",
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="edge-maxxing",
    )
