from __future__ import annotations

import logging
from importlib import util
from typing import Iterable, Literal, Tuple, Type, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T", bound=Type)
R = TypeVar("R")
R2 = TypeVar("R2")


def enumerate_v2(it: Iterable[R], start: int = 0) -> Iterable[Tuple[int, R]]:
    return enumerate(it, start=start)


def zip_v2(it1: Iterable[R], it2: Iterable[R2]) -> Iterable[Tuple[R, R2]]:
    return zip(it1, it2)


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
