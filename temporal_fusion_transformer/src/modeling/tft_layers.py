from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from temporal_fusion_transformer.src.modeling.self_attention import SelfAttention

if TYPE_CHECKING:
    from temporal_fusion_transformer.src.lib_types import ComputeDtype


class TimeDistributed(nn.Module):
    layer: nn.Module

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        input_shape = inputs.shape
        batch_size = input_shape[0]
        input_length = input_shape[1]
        inner_input_shape = input_shape[2:]
        # Shape: (num_samples * timesteps, ...). And track the
        # transformation in self._input_map.
        inputs = jnp.reshape(inputs, [-1, *inner_input_shape])
        # (num_samples * timesteps, ...)
        y = self.layer(inputs, *args, **kwargs)
        # Reconstruct the output shape by re-splitting the 0th dimension
        # back into (num_samples, timesteps, ...)
        # We use batch_size when available so that the 0th dimension is
        # set in the static shape of the reshaped output
        new_inner_shape = y.shape[1:]
        y = jnp.reshape(y, [batch_size, input_length, *new_inner_shape])
        return y


# -------------------------------------------------------------------------------------------------------------


class GatedLinearUnit(nn.Module):
    """
    Attributes
    ----------
    latent_dim:
        Latent space dimensionality.
    dropout_rate:
        Passed down to layers.Dropout(rate=dropout_rate)
    time_distributed:
        Apply across time axis yes/no.
    """

    latent_dim: int
    dropout_rate: float
    time_distributed: bool
    dtype: ComputeDtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training, name="dropout")(inputs)
        dense = nn.Dense(self.latent_dim, dtype=self.dtype, name="dense1")
        activation = nn.Sequential(
            [nn.Dense(self.latent_dim, dtype=self.dtype, name="dense2"), nn.sigmoid]
        )

        if self.time_distributed:
            dense = TimeDistributed(dense)
            activation = TimeDistributed(activation)

        x_pre_activation = dense(x)
        x_gated = activation(x)
        x = x_pre_activation * x_gated
        return x, x_gated


# -------------------------------------------------------------------------------------------------------------


class GatedResidualNetwork(nn.Module):
    """
    Attributes
    ----------
    latent_dim:
        Latent space dimensionality.
    dropout_rate:
        Dropout rate passed down to nn.Dropout.
    output_size:
        Size of output layer, default=latent_dim.
    time_distributed:
        Apply across time axis yes/no.

    References
    ----------

        - Bryan Lim, et.al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting:
            - https://arxiv.org/pdf/1912.09363.pdf
            - https://github.com/google-research/google-research/tree/master/tft
    """

    latent_dim: int
    dropout_rate: float
    time_distributed: bool
    output_size: int | None = None
    dtype: ComputeDtype = jnp.float32

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, context: jnp.ndarray | None = None, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.output_size is None:
            skip_connection = identity
        else:
            skip_connection = nn.Dense(self.output_size, dtype=self.dtype, name="skip")
            if self.time_distributed:
                skip_connection = TimeDistributed(skip_connection)

        pre_elu_dense = nn.Dense(self.latent_dim, dtype=self.dtype, name="dense_1")
        dense = nn.Dense(self.latent_dim, dtype=self.dtype, name="dense_2")

        if self.time_distributed:
            pre_elu_dense = TimeDistributed(pre_elu_dense)
            dense = TimeDistributed(dense)

        x_skip = skip_connection(inputs)
        x = pre_elu_dense(inputs)

        if context is not None:
            context_dense = nn.Dense(self.latent_dim, dtype=self.dtype, name="context")
            if self.time_distributed:
                context_dense = TimeDistributed(context_dense)
            x = x + context_dense(context)

        x = nn.elu(x)
        x = dense(x)
        x, gate = GatedLinearUnit(
            latent_dim=self.output_size or self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=self.time_distributed,
            dtype=self.dtype,
            name="glu",
        )(x, training=training)
        x = nn.LayerNorm(dtype=self.dtype, name="layer_norm")(x + x_skip)
        return x, gate


# -------------------------------------------------------------------------------------------------------------


class InputEmbedding(nn.Module):
    """
    This layer project all different inputs into the same latent space.
    Embedding is applied to categorical ones, and real-values ones are
    linearly mapped over `num_time_steps` axis.

    Attributes
    ----------

    static_categories_sizes:
        Sequence of ints describing with max value for each category of static inputs.
        E.g., you have `shop_name` (which can only be "a", "b" or "c") and `location_city`
        (which can only be "A", "B", "C" or "D") as your static inputs, then static_categories_size=[3, 4].

    known_categories_sizes:
        Sequence of ints describing with max value for each category of known categorical inputs.
        This follows the same principle as `static_categories_size`, with only difference that known inputs can
        change overtime. An example of know categorical input can be day of the week.

    latent_dim:
        Latent space dimensionality.

    num_known_real_inputs:

    num_observed_inputs:
    """

    static_categories_sizes: Sequence[int]
    known_categories_sizes: Sequence[int]
    num_known_real_inputs: int
    num_observed_inputs: int
    latent_dim: int

    num_unknown_inputs: int = 0
    dtype: ComputeDtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: InputStruct) -> EmbeddingStruct:
        static_input_embeddings = []
        known_real_inputs_embeddings = []

        for i, size in enumerate(self.static_categories_sizes):
            # Static are not time-varying, so we just take 1st time-step.
            embeds_i = nn.Embed(size, self.latent_dim, dtype=self.dtype, name=f"static_embed_{i}")(
                inputs.static[:, 0, i]
            )
            static_input_embeddings.append(embeds_i)

        static_input_embeddings = jnp.stack(static_input_embeddings, axis=1)
        for i in range(self.num_known_real_inputs):
            embeds_i = nn.Dense(self.latent_dim, dtype=self.dtype, name=f"known_dense_{i}")(
                inputs.known_real[..., i, jnp.newaxis]
            )
            known_real_inputs_embeddings.append(embeds_i)

        if self.num_observed_inputs != 0:
            observed_input_embeddings = []
            for i in range(self.num_observed_inputs):
                embeds_i = nn.Dense(self.latent_dim, dtype=self.dtype, name=f"observed_dense_{i}")(
                    inputs.observed[..., i, jnp.newaxis]
                )
                observed_input_embeddings.append(embeds_i)
            observed_input_embeddings = jnp.stack(observed_input_embeddings, axis=-1)
        else:
            observed_input_embeddings = None

        if len(self.known_categories_sizes) != 0:
            known_categorical_inputs_embeddings = []
            for i, size in enumerate(self.known_categories_sizes):
                embeds_i = nn.Embed(
                    size, self.latent_dim, dtype=self.dtype, name=f"known_embed_{i}"
                )(inputs.known_categorical[..., i])
                known_categorical_inputs_embeddings.append(embeds_i)
            known_inputs_embeddings = jnp.concatenate(
                [
                    jnp.stack(known_real_inputs_embeddings, axis=-1),
                    jnp.stack(known_categorical_inputs_embeddings, axis=-1),
                ],
                axis=-1,
            )

        else:
            known_inputs_embeddings = jnp.stack(known_real_inputs_embeddings, axis=-1)
        if self.num_unknown_inputs > 0:
            unknown = []
            for i in range(self.num_unknown_inputs):
                embeds_i = nn.Dense(self.latent_dim, dtype=self.dtype, name=f"unknown_dense_{i}")(
                    inputs.unknown[..., i, jnp.newaxis]
                )
                unknown.append(embeds_i)
            unknown = jnp.stack(unknown, axis=-1)
        else:
            unknown = None

        return EmbeddingStruct(
            known=known_inputs_embeddings,
            observed=observed_input_embeddings,
            static=static_input_embeddings,
            unknown=unknown,
        )


# -------------------------------------------------------------------------------------------------------------


class StaticCovariatesEncoder(nn.Module):

    """
    Create a static context out of static input embeddings.
    Static context is a (enrichment) vector, which must be added to other time varying inputs during
    `variable selection` process. Additionally, this layer creates initial state for LSTM cells,
    this way we give them `pre-memory`, and model the fact that static inputs do influence time dependent ones.

    Attributes
    ----------
    latent_dim:
        Dimensionality of the latent space.
    dropout_rate:
        Dropout rate passed down to keras.layer.Dropout.
    num_static_inputs
    """

    latent_dim: int
    dropout_rate: float
    num_static_inputs: int
    dtype: ComputeDtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool) -> StaticContextStruct:
        flat_x = flatten(inputs)

        mlp_outputs, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            output_size=self.num_static_inputs,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
            name="gru",
        )(flat_x, training=training)

        sparse_weights = nn.softmax(mlp_outputs)[..., jnp.newaxis]

        transformed_embeddings = []

        for i in range(self.num_static_inputs):
            embeds_i, _ = GatedResidualNetwork(
                latent_dim=self.latent_dim,
                dropout_rate=self.dropout_rate,
                time_distributed=False,
                dtype=self.dtype,
                name=f"gru_{i}",
            )(inputs[:, i][:, jnp.newaxis], training=training)
            transformed_embeddings.append(embeds_i)

        transformed_embeddings = jnp.concatenate(transformed_embeddings, axis=1)
        static_context_vector = jnp.sum(sparse_weights * transformed_embeddings, axis=1)
        context_variable_selection, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
            name="variable_selection",
        )(static_context_vector, training=training)
        context_enrichment, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
            name="enrichment",
        )(static_context_vector, training=training)
        context_state_h, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
            name="state_h",
        )(static_context_vector, training=training)
        context_state_c, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
            name="state_c",
        )(static_context_vector, training=training)
        return StaticContextStruct(
            enrichment=context_enrichment,
            state_h=context_state_h,
            state_c=context_state_c,
            vector=context_variable_selection,
            weight=sparse_weights,
        )


# -------------------------------------------------------------------------------------------------------------


class VariableSelectionNetwork(nn.Module):
    """
    Attributes
    ----------
    latent_dim:
        Latent space dimensionality.
    dropout_rate:
        Dropout rate passed down to keras.layer.Dropout.
    num_time_steps:
        Size of 2nd axis in the input.
    num_inputs:
        Size of 3rd axis in the input.
    """

    latent_dim: int
    dropout_rate: float
    num_time_steps: int
    num_inputs: int
    dtype: ComputeDtype = jnp.float32

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, context: jnp.ndarray, training: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mlp_outputs, static_gate = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            output_size=self.num_inputs,
            time_distributed=True,
            dtype=self.dtype,
            name="gru",
        )(
            jnp.reshape(inputs, [-1, self.num_time_steps, self.latent_dim * self.num_inputs]),
            context[:, jnp.newaxis],
            training=training,
        )
        sparse_weights = nn.softmax(mlp_outputs)
        sparse_weights = sparse_weights[:, :, jnp.newaxis]

        # Non-linear Processing & weight application
        transformed_embeddings = []
        for i in range(self.num_inputs):
            embeds_i, _ = GatedResidualNetwork(
                latent_dim=self.latent_dim,
                dropout_rate=self.dropout_rate,
                time_distributed=True,
                dtype=self.dtype,
                name=f"gru_{i}",
            )(inputs[..., i], training=training)
            transformed_embeddings.append(embeds_i)

        transformed_embeddings = jnp.stack(transformed_embeddings, axis=-1)
        temporal_ctx = jnp.sum(sparse_weights * transformed_embeddings, axis=-1)
        return temporal_ctx, sparse_weights, static_gate


# -------------------------------------------------------------------------------------------------------------


class DecoderBlock(nn.Module):
    """

    Attributes
    ----------
    num_attention_heads:
        Number of attention heads for MultiHeadSelfAttention.
    latent_dim:
        Latent space dimensionality.
    dropout_rate:
        Dropout rate passed down to keras.layer.Dropout.
    """

    num_attention_heads: int
    latent_dim: int
    dropout_rate: float
    attention_dropout_rate: float = 0.1
    dtype: ComputeDtype = jnp.float32

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, training: bool, mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        x = SelfAttention(
            num_heads=self.num_attention_heads,
            dtype=self.dtype,
            dropout_rate=self.attention_dropout_rate,
            # We probably can set decode=True during inference.
            # decode=not training,
            attention_fn=dot_product_attention,
            normalize_qk=True,
            name="attention",
        )(inputs, mask=mask, deterministic=not training)
        x = x.astype(self.dtype)
        x, _ = GatedLinearUnit(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
            name="glu",
        )(x, training=training)
        x = nn.LayerNorm(dtype=self.dtype, name="layer_norm")(x + inputs)
        # Nonlinear processing on outputs
        decoded, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
            name="gru",
        )(x, training=training)
        return decoded


# -------------------------------------------------------------------------------------------------------------


@struct.dataclass
class InputStruct:
    """
    Attributes
    ----------
    static:
        3D int
    known_real:
        3D float
    known_categorical:
        3D int
    observed:
        3D float
    unknown:
        3D float
    """

    static: jnp.ndarray
    known_real: jnp.ndarray
    known_categorical: jnp.ndarray | None
    observed: jnp.ndarray | None
    unknown: jnp.ndarray | None = None

    def cast_inexact(self, dtype: jnp.float32 | jnp.bfloat16 | jnp.float16) -> InputStruct:
        observed = self.observed
        unknown = self.unknown

        if observed is not None:
            observed = observed.astype(dtype)

        if unknown is not None:
            unknown = unknown.astype(dtype)

        return InputStruct(
            static=self.static,
            known_real=self.known_real.astype(dtype),
            known_categorical=self.known_categorical,
            unknown=unknown,
            observed=observed,
        )


@struct.dataclass
class EmbeddingStruct:
    """
    Attributes
    ----------

    static:
        3D float
    known:
        3D float
    observed:
        3D float
    unknown:
        3D float
    """

    static: jnp.ndarray
    known: jnp.ndarray
    observed: jnp.ndarray | None
    unknown: jnp.ndarray | None = None


@struct.dataclass
class StaticContextStruct:
    """
    Attributes
    ----------

    enrichment:
        has shape (batch_size, latent_dim) and must be used as additional context for GRNs.
    vector:
        has shape (batch_size, latent_dim), must be passed down to temporal decoder, and used  as additional context in GRNs.
    state_c:
        has shape (batch_size, latent_dim) and must be used together with `state_h` as initial context for LSTM cells.
    state_h
        has shape (batch_size, latent_dim) and must be used together with `state_c` as initial context for LSTM cells.
    """

    enrichment: jnp.ndarray
    state_h: jnp.ndarray
    state_c: jnp.ndarray
    vector: jnp.ndarray
    weight: jnp.ndarray


# -------------------------------------------------------------------------------------------------------------


@partial(jax.jit, inline=True)
def identity(arr: jnp.ndarray) -> jnp.ndarray:
    return jnp.array(arr)


@partial(jax.jit, inline=True)
def flatten(arr: jnp.ndarray) -> jnp.ndarray:
    """
    Flattens array preserving batch size
    Examples
    ----------

    >>> x = jnp.ones((8, 40, 5))
    >>> y = flatten(x)
    >>> y.shape
    (8, 200)

    """
    batch_size = arr.shape[0]
    new_arr = jnp.reshape(arr, (batch_size, -1))
    return new_arr


@partial(jax.jit, static_argnames=["dtype"], static_argnums=[1], inline=True)
def make_causal_mask(x: jnp.ndarray, dtype: ComputeDtype = jnp.float32) -> jnp.ndarray:
    return nn.make_causal_mask(x[..., 0], dtype=dtype)


@partial(
    jax.jit,
    inline=True,
    static_argnums=[5, 7, 8, 9, 10],
    static_argnames=["broadcast_dropout", "dropout_rate", "deterministic", "dtype", "precision"],
)
def dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    broadcast_dropout=True,
    dropout_rng=None,
    dropout_rate=0.0,
    deterministic=False,
    dtype=None,
    precision=None,
):
    """
    We force attention computations to be in f32 to avoid f16 overflow.
    After we cast it back to main compute dtype.
    """
    attention_dtype = jnp.float32 if dtype == jnp.float16 else dtype

    return nn.attention.dot_product_attention(
        query=query,
        key=key,
        value=value,
        bias=bias,
        mask=mask,
        broadcast_dropout=broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dtype=attention_dtype,
        precision=precision,
    ).astype(dtype)
