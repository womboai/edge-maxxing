import json
import pprint
import sys
from dataclasses import dataclass
from os import makedirs
from os.path import expanduser, join, isfile
from pathlib import Path

import bittensor as bt
from torch import load

from validator import ContestState  # noqa

PATH: Path = Path(".diagnostics.json")


@dataclass
class DiagnosticsData:
    process_command: str
    wallet_name: str
    wallet_hotkey: str
    netuid: int
    logging_dir: str


def state_path(data: DiagnosticsData) -> str:
    full_path = expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            data.logging_dir,
            data.wallet_name,
            data.wallet_hotkey,
            data.netuid,
            "validator",
        )
    )

    makedirs(full_path, exist_ok=True)

    return join(full_path, "state.pt")


def load_state(diagnostics: DiagnosticsData) -> dict:
    path = state_path(diagnostics)

    if not isfile(path):
        return {}

    return load(path)


def save_validator_diagnostics(config: bt.config):
    data = {
        "process_command": " ".join(sys.argv),
        "wallet": {
            "name": config.wallet.name,
            "hotkey": config.wallet.hotkey,
        },
        "netuid": config.netuid,
        "logging": {
            "logging_dir": config.logging.logging_dir,
        },
    }

    with open(PATH, "w") as f:
        json.dump(data, f, indent=4)


def load_validator_diagnostics() -> DiagnosticsData:
    if not isfile(PATH):
        exit("No diagnostics file found")

    with open(PATH, "r") as f:
        data = json.load(f)

        return DiagnosticsData(
            process_command=data["process_command"],
            wallet_name=data["wallet"]["name"],
            wallet_hotkey=data["wallet"]["hotkey"],
            netuid=data["netuid"],
            logging_dir=data["logging"]["logging_dir"],
        )


if __name__ == "__main__":
    diagnostics = load_validator_diagnostics()
    print("Process Command:")
    print(diagnostics.process_command)
    print("Validator State:")
    data = load_state(diagnostics)
    pprint.pprint(data)
