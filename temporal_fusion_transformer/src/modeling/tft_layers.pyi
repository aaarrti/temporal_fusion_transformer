from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict
from keras import layers
import tensorflow as tf

class InputEmbedding(layers.Layer):
    def __init__(
        self,
        static_categories_sizes: Sequence[int],
        known_categories_sizes: Sequence[int],
        input_observed_idx: Sequence[int],
        input_static_idx: Sequence[int],
        input_known_real_idx: Sequence[int],
        input_known_categorical_idx: Sequence[int],
        hidden_layer_size: int,
        **kwargs,
    ): ...
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor: ...

class TransformerBlock(layers.Layer):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_layer_size: int,
        dropout_rate: float,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ): ...
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor: ...

# -------------------------------------------------------------------------------------------------------------

class Linear(layers.Layer):
    """
    Args:
      size: Output size
      activation: Activation function to apply if required
      use_time_distributed: Whether to apply layer across time
      use_bias: Whether bias should be included in layer
    """

    def __init__(
        self,
        size: int,
        activation: str | None = None,
        use_time_distributed: bool = False,
        use_bias: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ): ...
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor: ...

class GatedLinearUnit(layers.Layer):
    """Applies a Gated Linear Unit (GLU) to an input.

    Args:
      x: Input to gating layer
      hidden_layer_size: Dimension of GLU
      dropout_rate: Dropout rate to apply if any
      use_time_distributed: Whether to apply across time
      activation: Activation function to apply to the linear feature transform if
        necessary

    Returns:
      Tuple of tensors for: (GLU output, gate)
    """

    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate: float | None = None,
        use_time_distributed: bool = True,
        activation: str | None = None,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ): ...
    def __call__(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]: ...

class AddAndNorm(layers.Layer):
    """
    Applies skip connection followed by layer normalisation.

    Args:
      x_Sequence: Sequence of inputs to sum for skip connection

    Returns:
        Tensor output from layer.
    """

    def __init__(self, **kwargs): ...
    def __call__(self, x: tf.Tensor) -> tf.Tensor: ...

class GatedResidualNetwork(layers.Layer):
    """Applies the gated residual network (GRN) as defined in paper.

      Args:
    x: Network inputs
    hidden_layer_size: Internal state size
    output_size: Size of output layer
    dropout_rate: Dropout rate if dropout is applied
    use_time_distributed: Whether to apply network across time dimension
    additional_context: Additional context vector to use if relevant
    return_gate: Whether to return GLU gate for diagnostic purposes

      Returns:
    Tuple of tensors for: (GRN output, GLU gate)
    """

    def __init__(
        self,
        hidden_layer_size: int,
        output_size: int | None = None,
        dropout_rate: float | None = None,
        use_time_distributed: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ): ...
    def __call__(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]: ...

class ContextInput(TypedDict):
    input: tf.Tensor
    context: tf.Tensor

class GatedResidualNetworkWithContext(GatedResidualNetwork):
    skip: layers.Layer
    hidden: layers.Layer
    elu: layers.Layer
    hidden_2: layers.Layer
    glu: layers.Layer
    add_and_norm: layers.Layer

    def __call__(self, x: ContextInput) -> tuple[tf.Tensor, tf.Tensor]: ...

class StaticVariableSelectionNetwork(layers.Layer):
    """Applies variable selection network to static inputs.

    Args:
      embedding: Transformed static inputs
      hidden_layer_size
      dropout_rate
      num_static

    Returns:
      Tensor output for variable selection network
    """

    def __init__(
        self,
        num_static: int,
        hidden_layer_size: int,
        dropout_rate: float,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ): ...
    def __call__(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]: ...

class VariableSelectionNetwork(layers.Layer):
    """Apply temporal variable selection networks.

    Args:
      embedding: Transformed inputs.
      static_context_variable_selection

      dropout_rate
      hidden_layer_size

    Returns:
      Processed tensor outputs.
    """

    def __init__(
        self,
        dropout_rate: float,
        hidden_layer_size: int,
        num_inputs: int,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ): ...
    def __call__(self, x: ContextInput) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]: ...
