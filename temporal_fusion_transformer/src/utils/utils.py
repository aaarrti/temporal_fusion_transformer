from __future__ import annotations

from typing import Dict

from jax.tree_util import tree_flatten_with_path


class classproperty(property):
    def __get__(self, _, owner_cls: Type):  # noqa
        return self.fget(owner_cls)


def flatten_dict(d: Dict[str, ...]) -> Dict[str, ...]:
    accumulator = {}
    flat_tree = tree_flatten_with_path(d, lambda i: not isinstance(i, Dict))[0]

    for k, v in flat_tree:
        k = k[-1].key
        if k in accumulator:
            raise RuntimeError(f"Found duplicate key: {k}")
        accumulator[k] = v

    return accumulator
