from argparse import ArgumentParser
from typing import TypeVar

from bittensor import config

from . import Neuron

T: TypeVar = TypeVar("T", bound=Neuron)


def get_config(cls: type[T]):
    argument_parser = ArgumentParser()

    cls.add_args(argument_parser)

    return config(argument_parser)
