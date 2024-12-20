from typing import Callable, TypeVar

from fiber.chain.interface import get_substrate
from substrateinterface import SubstrateInterface
from tenacity import retry, stop_after_attempt, wait_exponential

R = TypeVar("R")


class SubstrateHandler:
    _substrate: SubstrateInterface

    def __init__(self, url: str):
        self._substrate = get_substrate(url)

    @property
    def substrate(self):
        return self._substrate

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=5),
        reraise=True,
    )
    def execute(self, action: Callable[[SubstrateInterface], R]):
        try:
            return action(self._substrate)
        except BaseException:
            self._substrate.close()

            raise
