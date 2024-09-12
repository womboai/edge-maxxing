from typing import Generator
import logging


def __all_loggers() -> Generator[logging.Logger, None, None]:
    """Generator that yields all logger instances in the application.

    Iterates through the logging root manager's logger dictionary and yields all active `Logger` instances. It skips
    placeholders and other types that are not instances of `Logger`.

    Yields:
        logger (logging.Logger): An active logger instance.
    """
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.PlaceHolder):
            continue
        # In some versions of Python, the values in loggerDict might be
        # LoggerAdapter instances instead of Logger instances.
        # We check for Logger instances specifically.
        if isinstance(logger, logging.Logger):
            yield logger
        else:
            # If it's not a Logger instance, it could be a LoggerAdapter or
            # another form that doesn't directly offer logging methods.
            # This branch can be extended to handle such cases as needed.
            pass


__cached_handlers = {}

for __logger in __all_loggers():
    __cached_handlers[__logger] = __logger.handlers.copy()

from bittensor import *
from bittensor.extrinsics.serving import *

for __logger in __all_loggers():
    if __logger not in __cached_handlers:
        continue

    for handler in __cached_handlers[__logger]:
        __logger.addHandler(handler)
