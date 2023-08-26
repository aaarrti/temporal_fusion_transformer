from clu.metrics import Collection, Average
from jaxtyping import jaxtyped
from flax.struct import dataclass


@jaxtyped
@dataclass
class MetricContainer(Collection):
    """We define this simple container in separate file, because it is not compatible
    with `from __future__ import annotations`."""

    loss: Average.from_output("loss")
