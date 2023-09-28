from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax.linen import initializers
from flax.linen.attention import combine_masks, dot_product_attention
from flax.linen.linear import (
    DenseGeneral,
    DotGeneralT,
    PrecisionLike,
    default_kernel_init,
)
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import LayerNorm
from jax import lax
from jax.random import PRNGKey

Shape = Tuple[int, ...]
Dtype = Any
Array = jnp.ndarray

# For python 3.8, `normalize_qk` is not available, so we just copy-paste if from newer source code.
# Can be removed after Kaggle VM is upgrade to support flax 0.3.3+


class MultiHeadDotProductAttention(Module):
    num_heads: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    qkv_features: int | None = None
    out_features: int | None = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    deterministic: bool | None = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    use_bias: bool = True
    attention_fn: Callable[..., Array] = dot_product_attention
    decode: bool = False
    normalize_qk: bool = False
    # Deprecated, will be removed.
    qkv_dot_general: DotGeneralT = lax.dot_general
    out_dot_general: DotGeneralT = lax.dot_general

    @compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Array | None = None,
        deterministic: bool | None = None,
    ):
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            f"Memory dimension ({qkv_features}) must be divisible by number of" f" heads ({self.num_heads})."
        )
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (
            dense(name="query")(inputs_q),
            dense(name="key")(inputs_kv),
            dense(name="value")(inputs_kv),
        )

        if self.normalize_qk:
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = LayerNorm(name="query_ln", use_bias=False)(query)  # type: ignore[call-arg]
            key = LayerNorm(name="key_ln", use_bias=False)(key)  # type: ignore[call-arg]

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.decode:
            # detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable("cache", "cached_key")
            cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
            cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
            cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
            if is_initialized:
                (
                    *batch_dims,
                    max_length,
                    num_heads,
                    depth_per_head,
                ) = cached_key.value.shape
                # shape check of cached keys against query input
                expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        "expected query shape %s instead got %s." % (expected_shape, query.shape)
                    )
                # update key, value caches with our new 1d spatial slices
                cur_index = cache_index.value
                indices: tuple[Union[int, jax.Array], ...] = (0,) * len(batch_dims) + (
                    cur_index,
                    0,
                    0,
                )
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # causal mask for cached decoder self-attention:
                # our single query position should only attend to those key
                # positions that have already been generated and cached,
                # not the remaining zero elements.
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(max_length) <= cur_index,
                        tuple(batch_dims) + (1, 1, max_length),
                    ),
                )

        dropout_rng = None
        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            m_deterministic = merge_param("deterministic", self.deterministic, deterministic)
            if not m_deterministic:
                dropout_rng = self.make_rng("dropout")
        else:
            m_deterministic = True

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=m_deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )  # pytype: disable=wrong-keyword-args
        # back to the original inputs dimensions
        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=self.out_dot_general,
            name="out",  # type: ignore[call-arg]
        )(x)
        return out


class SelfAttention(MultiHeadDotProductAttention):
    @compact
    def __call__(  # type: ignore
        self,
        inputs_q: Array,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
    ):
        return super().__call__(inputs_q, inputs_q, mask, deterministic=deterministic)
