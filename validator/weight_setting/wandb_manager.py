from argparse import ArgumentParser
from typing import Any

import wandb
from wandb.apis.public import Run

from base.checkpoint import Uid, Key
from base.system_info import SystemInfo
from .contest_state import ContestState


class WandbManager:
    _run: Run | None = None

    config: dict[str, Any]
    validator_version: str
    uid: Uid
    netuid: Uid
    hotkey: str
    signature: str

    def __init__(
        self,
        config: dict[str, Any],
        validator_version: str,
        uid: Uid,
        netuid: Uid,
        hotkey: str,
        signature: str,
    ):
        self.config = config
        self.validator_version = validator_version
        self.uid = uid
        self.netuid = netuid
        self.hotkey = hotkey
        self.signature = signature

    def init_wandb(self, contest_state: ContestState):
        if self.config["wandb.off"]:
            return

        if self._run:
            self._run.finish()

        day = contest_state.get_contest_start()
        name = f"validator-{self.uid}-{day.year}-{day.month}-{day.day}"

        self._run = wandb.init(
            name=name,
            id=name,
            resume="allow",
            mode="offline" if self.config["wandb.offline"] else "online",
            project=self.config["wandb.project_name"],
            entity=self.config["wandb.entity"],
            notes=self.config["wandb.notes"],
            config={
                "hotkey": self.hotkey,
                "type": "validator",
                "uid": self.uid,
                "signature": self.signature,
            },
            allow_val_change=True,
            anonymous="allow",
            tags=[
                f"version_{self.validator_version}",
                f"sn{self.netuid}",
            ],
        )

    def send_metrics(
        self,
        contest_state: ContestState,
        api_hardware: list[SystemInfo],
        scores: dict[Key, float] | None = None,
        ranks: dict[Key, int] | None = None
    ):
        if not self._run or self.config["wandb.off"]:
            return

        data = {
            "scores": scores or contest_state.get_scores(contest_state.benchmarks),
            "api_hardware": [api.model_dump() for api in api_hardware],
            "ranks": ranks or contest_state.get_ranks(scores),
            "num_gpus": len(self.config["benchmarker_api"]),
        } | contest_state.model_dump()

        self._run.log(data=data)


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
