import tensorflow as tf

class FusedScaleMaskSoftmax(tf.keras.layers.Layer):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        attn_mask_type: attention mask type (pad or causal)
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        attn_mask_type: str,
        mask_func,
        softmax_in_fp32: bool,
        scale: float,
    ): ...
