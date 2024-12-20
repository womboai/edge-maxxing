from typing import Callable, TypeVar

from substrateinterface import SubstrateInterface
from tenacity import retry, stop_after_attempt, wait_exponential

R = TypeVar("R")


class SubstrateHandler:
    _substrate: SubstrateInterface

    def __init__(self, substrate: SubstrateInterface):
        self._substrate = substrate

    @property
    def substrate(self):
        return self._substrate

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=5),
        reraise=True,
    )
    def execute(self, action: Callable[[SubstrateInterface], R]) -> R:
        try:
            return action(self._substrate)
        except BaseException:
            self._substrate.close()

            raise
