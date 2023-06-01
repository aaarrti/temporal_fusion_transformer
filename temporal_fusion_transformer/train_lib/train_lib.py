from typing import NamedTuple


class HyperParameters(NamedTuple):
    """
    max_gradient_norm: Maximum norm for gradient clipping
    learning_rate: Initial learning rate of ADAM optimizer
    batch_size: Size of batches for training
    num_epochs: Maximum number of epochs for training
    """

    max_gradient_norm: float
    num_epochs: float
    learning_rate: float
    batch_size: float
