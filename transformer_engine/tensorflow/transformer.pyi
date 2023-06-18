import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Parallel attention w/ QKV and Proj Gemms
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float,
        layernorm_epsilon: float = 1e-3,
        init_method=None,
        output_layer_init_method=None,
        layer_number=None,
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = False,
        attn_mask_type: str = "causal",
        return_layernorm_output: bool = False,
        input_layernorm: bool = False,
        attention_type: str = "self",
        fuse_qkv_params: bool = False,
    ): ...

class DropPath(tf.keras.Model):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0): ...

class TransformerLayer(tf.keras.Model):
    """
    TransformerLayer is made up of an attention block and a feedforward network
    (MLP). This standard layer is based on the paper
    "Attention Is All You Need".

    Parameters
    ----------
    hidden_size : int
      size of each input sample.
    ffn_hidden_size : int
      intermediate size to which input samples are projected.
    num_attention_heads : int
      number of attention heads in the transformer layer.
    layernorm_epsilon : float, default = 1e-5
      a value added to the denominator of layer normalization for numerical
      stability.
    hidden_dropout: float, default = 0.1
      dropout probability for the dropout op after FC2 layer.
    attention_dropout: float, default = 0.1
      dropout probability for the dropout op during multi-head attention.
    init_method : Callable, default = `None`
      used for initializing weights of QKV and FC1 weights in the following way:
      `init_method(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
      used for initializing weights of PROJ and FC2 in the following way:
      `output_layer_init_method(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    apply_residual_connection_post_layernorm : bool, default = `False`
      if set to `True`, residual connections are taken from the output of layer
      norm (default is taken from input of layer norm)
    layer_number: int, default = `None`
      layer number of the current `TransformerLayer` when multiple such modules
      are concatenated to form a transformer block.
    apply_query_key_layer_scaling: bool, default = `True`
      apply query-key layer scaling during BMM1 by a factor of `layer_number`
    output_layernorm: bool, default = `False`
      if set to `True`, layer normalization is applied on the output side, after
      the final dropout-add. default behavior is to apply layer normalization on
      the input side, before the QKV transformation.
    attention_softmax_in_fp32: bool, default = `False`
      if set to `True`, softmax is executed in tf.float32 dtype (single
      precision)
    layer_type: {'encoder', 'decoder'}, default = `encoder`
      if set to `decoder`, an additional cross-attn block is added after
      self-attn. This can be used for structures like `T5` Transformer in
      conjunction with the `encoder` option.
    kv_channels: int, default = `None`
      number of key-value channels. defaults to
      `hidden_size / num_attention_heads` if `None`.
    self_attn_mask_type: {'causal', 'padding'}, default = `causal`
      type of attention mask passed into softmax operation.

    Optimization parameters
    -----------------------
    drop_path_rate: float, default = 0.0
      when > 0.0, applies stochastic depth per sample in the main path of the
      residual block.
    fuse_qkv_params: bool, default = 'False'
      if set to `True`, `TransformerLayer` module exposes a single fused
      parameter for query-key-value. This enables optimizations such as QKV
      fusion without concatentations/splits.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        init_method=None,
        output_layer_init_method=None,
        layer_number=None,
        kv_channels=None,
        self_attn_mask_type: str = "causal",
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = False,
        apply_residual_connection_post_layernorm: bool = False,
        output_layernorm: bool = False,
        layer_type: str = "encoder",
        drop_path_rate: float = 0.0,
        fuse_qkv_params: bool = False,
    ): ...
