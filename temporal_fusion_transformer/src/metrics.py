from clu.metrics import Average, Collection
from flax.struct import dataclass
from jaxtyping import jaxtyped

# TODO: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/metrics/r_square.py


@jaxtyped
@dataclass
class MetricContainer(Collection):
    """We define this simple container in separate file, because it is not compatible
    with `from __future__ import annotations`."""

    loss: Average.from_output("loss")
