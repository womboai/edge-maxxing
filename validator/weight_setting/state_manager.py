from pathlib import Path

from fiber.logging_utils import get_logger
from pydantic_core import ValidationError

from .contest_state import ContestState

logger = get_logger(__name__)

STATE_VERSION = 7


class StateManager:
    _state_file: Path

    def __init__(self, wallet_name: str, hotkey_name: str, netuid: int):
        path = (
            Path.home() /
            ".bittensor" /
            "miners" /
            wallet_name /
            hotkey_name /
            f"netuid{netuid}" /
            "validator"
        )

        path.mkdir(parents=True, exist_ok=True)
        self._state_file = path / f"state_v{STATE_VERSION}.json"

    def load_state(self) -> ContestState | None:
        if not self._state_file.exists():
            return None

        logger.info(f"Loading state")

        try:
            with self._state_file.open("rb") as file:
                return ContestState.model_validate_json(file.read())
        except ValidationError as e:
            logger.error(f"Failed to load state", exc_info=e)
            return None

    def save_state(self, state: ContestState):
        logger.debug(f"Saving state")

        with self._state_file.open("wb") as file:
            file.write(state.model_dump_json(indent=4).encode())
