import json
import pprint
import shutil
import sys
import zipfile
from dataclasses import dataclass
from os import makedirs
from os.path import expanduser, join, isfile
from pathlib import Path

import bittensor as bt
from torch import load

from validator import ContestState  # noqa (Needed for depickling)

DIAGNOSTICS_DIR: Path = Path(".diagnostics")
DIAGNOSTICS_FILE_PATH: Path = DIAGNOSTICS_DIR / "diagnostics.json"


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

    DIAGNOSTICS_DIR.mkdir(exist_ok=True)
    with open(DIAGNOSTICS_FILE_PATH, "w") as f:
        json.dump(data, f, indent=4)


def load_validator_diagnostics() -> DiagnosticsData:
    if not isfile(DIAGNOSTICS_FILE_PATH):
        exit("No diagnostics file found")

    with open(DIAGNOSTICS_FILE_PATH, "r") as f:
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

    print("Gathering validator state")
    data = load_state(diagnostics)
    with open(DIAGNOSTICS_DIR / "state.txt", 'w') as file:
        pprint.pprint(data, stream=file)

    print("Gathering logs")
    logs = Path(expanduser("~/.pm2/logs"))
    for file in logs.iterdir():
        shutil.copy(file, DIAGNOSTICS_DIR)

    with zipfile.ZipFile("diagnostics.zip", "w") as zf:
        for file in DIAGNOSTICS_DIR.iterdir():
            zf.write(file)

    print(f"Exported diagnostics to {Path.cwd()}/diagnostics.zip")

    shutil.rmtree(DIAGNOSTICS_DIR, ignore_errors=True)
