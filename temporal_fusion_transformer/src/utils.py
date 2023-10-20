from __future__ import annotations

import logging
from importlib import util
from typing import Callable, Iterable, Literal, Tuple, Type, TypeVar, no_type_check

log = logging.getLogger(__name__)
T = TypeVar("T", bound=Type)
R = TypeVar("R")


class classproperty(property):
    def __init__(self, fget=Callable[[T], R]):
        super().__init__(fget=fget)

    @no_type_check
    def __get__(self, _, owner_cls: T) -> R:
        return self.fget(owner_cls)


def enumerate_v2(it: Iterable[R], start: int = 0) -> Iterable[Tuple[int, R]]:
    return enumerate(it, start=start)


def setup_logging(
    *,
    log_format: str = "%(asctime)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s",
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "DEBUG",
):
    import logging

    import absl.logging

    logging.basicConfig(
        level=logging.getLevelName(log_level),
        format=log_format,
    )

    absl.logging.set_verbosity(absl.logging.converter.ABSL_NAMES[log_level])

    if util.find_spec("tensorflow"):
        import tensorflow as tf

        tf.get_logger().setLevel(log_level)
