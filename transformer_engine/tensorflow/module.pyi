from typing import Union, Callable

from keras import layers

# TransformerEngineBaseModule is a mixin class and its init function will pass
# through all the positional and keyword arguments to other subclasses. Make
# sure this class is inherited first.
class TransformerEngineBaseModule:
    """Base TE module."""

    # fp8 related
    fp8: bool
    fp8_meta: dict[str, str]
    fp8_meta_tensors_initialized: bool
    fp8_weight_shapes: list
    stream_id: str

class Dense(TransformerEngineBaseModule, layers.Layer):
    """
    Applies a linear transformation to the incoming data :math:`y = xW + b`

    On NVIDIA GPUs it is a drop-in replacement for `tf.keras.layers.Dense`.

    Parameters
    ----------
    units : int
      size of each output sample.
    use_bias : bool, default = `True`
      if set to `False`, the layer will not learn an additive bias.
    kernel_initializer: Callable, default = `None`
      used for initializing weights in the following way:
      `kernel_initializer(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    bias_initializer: Callable, default = `None`
      used for initializing biases in the following way:
      `bias_initializer(weight)`. When set to `None`, defaults to `zeros`.

    Parallelism parameters
    ----------------------
    skip_weight_param_allocation: bool, default = `False`
      if set to `True`, weight parameter is not allocated and must be passed as
      a keyword argument `weight` during the forward pass.

    Optimization parameters
    -----------------------
    return_bias : bool, default = `False`
      when set to `True`, this module will not apply the additive bias itself,
      but instead return the bias value during the forward pass together with
      the output of the linear transformation :math:`y = xW`. This is useful
      when the bias addition can be fused to subsequent operations.
    """

    def __init__(
        self,
        units: int,
        use_bias: bool = True,
        return_bias: bool = False,
        kernel_initializer: Union[Callable, str, None] = None,
        bias_initializer: Union[Callable, str, None] = None,
        skip_weight_param_allocation: bool = False,
        **kwargs,
    ): ...

class LayerNorm(layers.Layer):
    """
    Applies Layer Normalization over a mini-batch of inputs.

    Parameters
    ----------
    epsilon : float, default = 1e-3
      a value added to the denominator of layer normalization for numerical
      stability.
    gamma_initializer: Callable, default = `None`
      used for initializing LayerNorm gamma in the following way:
      `gamma_initializer(weight)`. When set to `None`, defaults to `ones`.
    beta_initializer: Callable, default = `None`
      used for initializing LayerNorm beta in the following way:
      `beta_initializer(weight)`. When set to `None`, defaults to `zeros`.
    """

    def __init__(
        self, epsilon=1e-3, gamma_initializer="ones", beta_initializer="zeros", **kwargs
    ): ...

class LayerNormDense(TransformerEngineBaseModule, layers.Layer):
    """
    Applies layer normalization followed by linear transformation to the
    incoming data.

    Parameters
    ----------
    units : int
      size of each output sample.
    epsilon : float, default = 1e-3
      a value added to the denominator of layer normalization for numerical
      stability.
    use_bias : bool, default = `True`
      if set to `False`, the layer will not learn an additive bias.
    gamma_initializer: Callable, default = `None`
      used for initializing LayerNorm gamma in the following way:
      `gamma_initializer(weight)`. When set to `None`, defaults to `ones`.
    beta_initializer: Callable, default = `None`
      used for initializing LayerNorm beta in the following way:
      `beta_initializer(weight)`. When set to `None`, defaults to `zeros`.
    kernel_initializer : Callable, default = `None`
      used for initializing GEMM weights in the following way:
      `kernel_initializer(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    bias_initializer : Callable, default = `None`
      used for initializing GEMM bias in the following way:
      `bias_initializer(weight)`. When set to `None`, defaults to `zeros`.
    return_layernorm_output : bool, default = `False`
      if set to `True`, output of layernorm is returned from the forward
      together with the output of the linear transformation.
      Example use case: residual connection for transformer module is taken post
      layernorm.

    Parallelism parameters
    ----------------------
    skip_weight_param_allocation: bool, default = `False`
      if set to `True`, weight parameter is not allocated and must be passed as
      a keyword argument `weight` during the forward pass.

    Optimization parameters
    -----------------------
    return_bias : bool, default = `False`
      when set to `True`, this module will not apply the additive bias itself,
      but instead return the bias value during the forward pass together with
      the output of the linear transformation :math:`y = xW`. This is useful
      when the bias addition can be fused to subsequent operations.
    """

    def __init__(
        self,
        units,
        epsilon=1e-3,
        gamma_initializer: Union[Callable, str, None] = None,
        beta_initializer: Union[Callable, str, None] = None,
        return_layernorm_output=False,
        use_bias=True,
        return_bias=False,
        kernel_initializer: Union[Callable, str, None] = None,
        bias_initializer: Union[Callable, str, None] = None,
        skip_weight_param_allocation=False,
        **kwargs,
    ): ...

class LayerNormMLP(TransformerEngineBaseModule, layers.Layer):
    """
    Applies layer normalization on the input followed by the MLP module,
    consisting of 2 successive linear transformations, separated by the GeLU
    activation.

    Parameters
    ----------
    units : int
      size of each input sample.
    ffn_units : int
      intermediate size to which input samples are projected.
    epsilon : float, default = 1e-3
      a value added to the denominator of layer normalization for numerical
      stability.
    gamma_initializer: Callable, default = `None`
      used for initializing LayerNorm gamma in the following way:
      `gamma_initializer(weight)`. When set to `None`, defaults to `ones`.
    beta_initializer: Callable, default = `None`
      used for initializing LayerNorm beta in the following way:
      `beta_initializer(weight)`. When set to `None`, defaults to `zeros`.
    use_bias : bool, default = `True`
      if set to `False`, the FC2 layer will not learn an additive bias.
    kernel_initializer: Callable, default = `None`
      used for initializing FC1 weights in the following way:
      `kernel_initializer(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    ffn_kernel_initializer: Callable, default = `None`
      used for initializing FC2 weights in the following way:
      `ffn_kernel_initializer(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
      if set to `True`, output of layernorm is returned from the forward
      together with the output of the linear transformation.
      Example use case: residual connection for transformer module is taken post
      layernorm.
    bias_initializer: Callable, default = `None`
      used for initializing FC1 and FC2 bias in the following way:
      `bias_initializer(weight)`. When set to `None`, defaults to `zeros`.

    Optimization parameters
    -----------------------
    return_bias : bool, default = `False`
      when set to `True`, this module will not apply the additive bias itself,
      but instead return the bias value during the forward pass together with
      the output of the linear transformation :math:`y = xW`. This is useful
      when the bias addition can be fused to subsequent operations.
    """

    def __init__(
        self,
        units: int,
        ffn_units: int,
        epsilon: float = 1e-3,
        gamma_initializer: Union[Callable, str, None] = None,
        beta_initializer: Union[Callable, str, None] = None,
        return_layernorm_output: bool = False,
        use_bias: bool = True,
        return_bias: bool = False,
        kernel_initializer: Union[Callable, str, None] = None,
        ffn_kernel_initializer: Union[Callable, str, None] = None,
        bias_initializer: Union[Callable, str, None] = None,
        **kwargs,
    ): ...
