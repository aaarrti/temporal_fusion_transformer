from __future__ import annotations

import functools
import inspect
import logging
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from importlib import util
from types import FunctionType, MethodType
from typing import Any, Literal, TypeVar

import tomli
import toolz

from temporal_fusion_transformer.src.config import Config

log = logging.getLogger(__name__)
T = TypeVar("T", bound=type)
R = TypeVar("R")
R2 = TypeVar("R2")
C = TypeVar("C", bound=Callable)


def enumerate_v2(it: Iterable[R], start: int = 0) -> Iterable[tuple[int, R]]:
    return enumerate(it, start=start)


def zip_v2(it1: Iterable[R], it2: Iterable[R2]) -> Iterable[tuple[R, R2]]:
    return zip(it1, it2)


def dict_map(d: dict[str, R], map_fn: Callable[[R], R2]) -> dict[str, R2]:
    new_dict = {}

    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_map(v, map_fn)
        else:
            v = map_fn(v)
        new_dict[k] = v

    return new_dict


@toolz.curry
def log_before(
    func: C,
    logger: Callable[[str], None] = log.debug,
    ignore_argnums: Sequence[int] = (),
    ignore_argnames: Sequence[str] = (),
) -> C:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = format_callable_args(func_args, ignore_argnums, ignore_argnames)
        func_name_str = format_callable_name(func)
        logger(f"Entered {func_name_str} with args ( {func_args_str} )")
        return func(*args, **kwargs)

    return wrapper


@toolz.curry
def log_after(func: C, logger: Callable[[str], None] = log.debug) -> C:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        retval = func(*args, **kwargs)
        func_name_str = format_callable_name(func)
        logger(f"Exited {func_name_str}(...) with value: {repr(retval)}")
        return retval

    return wrapper


def format_callable_name(func: Callable) -> str:
    if hasattr(func, "__wrapped__"):
        return format_callable_name(func.__wrapped__)

    if isinstance(func, functools.partial):
        return f"partial({format_callable_name(func.func)})"

    if inspect.isfunction(func):
        _func: FunctionType = func
        return f"{_func.__module__}.{_func.__qualname__}"

    elif inspect.ismethod(func):
        _method: MethodType = func
        return f"{_method.__module__}.{_method.__class__}.{_method.__qualname__}"

    else:
        log.error(f"Don't know how to format name of ${func}")
        return repr(func)


def format_callable_args(
    arguments: OrderedDict[str, Any],
    ignore_argnums: Sequence[int] = (),
    ignore_argnames: Sequence[str] = (),
) -> str:
    filtered_args = {}

    for i, (k, v) in enumerate(arguments.items()):
        if i not in ignore_argnums and k not in ignore_argnames:
            filtered_args[k] = v

    return ", ".join(map("{0[0]} = {0[1]!r}".format, filtered_args.items()))


def make_timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def count_inputs(config: Config) -> int:
    return (
        len(config.input_observed_idx)
        + len(config.input_static_idx)
        + len(config.input_known_real_idx)
        + len(config.input_known_categorical_idx)
    )
