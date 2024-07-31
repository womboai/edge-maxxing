import os
from argparse import ArgumentParser
import bittensor as bt


def add_wandb_args(parser: ArgumentParser):
    wandb_off = not os.getenv("WANDB_API_KEY")

    if wandb_off:
        bt.logging.warning(
            "WANDB_API_KEY not set, running without wandb. "
            "We highly recommend setting WANDB_API_KEY to enable W&B."
        )

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=wandb_off,
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
