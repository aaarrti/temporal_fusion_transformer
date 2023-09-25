from clu.metrics import Average, Collection
from flax.struct import dataclass


@dataclass
class MetricContainer(Collection):
    """We define this simple container in separate file, because it is not compatible
    with `from __future__ import annotations`."""

    loss: Average.from_output("loss")
