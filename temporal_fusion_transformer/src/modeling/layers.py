from __future__ import annotations

import functools
from typing import Sequence, Type, TypeVar, Any, TypeAlias, Union, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct
from jax.typing import DTypeLike

ComputeDtype: TypeAlias = Union[
    jnp.float32, jnp.float16, jnp.bfloat16, Literal["float32", "float16", "bfloat16"], Any
]
T = TypeVar("T", bound=Type[nn.Module])


class TimeDistributed(nn.Module):
    layer: nn.Module

    @nn.compact
    def __call__(self, inputs: jax.Array, **kwargs) -> jax.Array:
        input_shape = inputs.shape
        batch_size = input_shape[0]
        input_length = input_shape[1]
        inner_input_shape = input_shape[2:]
        # Shape: (num_samples * timesteps, ...). And track the
        # transformation in self._input_map.
        inputs = jnp.reshape(inputs, [-1, *inner_input_shape])
        # (num_samples * timesteps, ...)
        y = self.layer(inputs, **kwargs)
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
    def __call__(self, inputs: jax.Array, training: bool) -> tuple[jax.Array, jax.Array]:
        x = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=not training,
        )(inputs)
        dense = nn.Dense(self.latent_dim, dtype=self.dtype)
        activation = nn.Sequential([nn.Dense(self.latent_dim, dtype=self.dtype), nn.sigmoid])

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
        Dropout rate passed down to keras.layer.Dropout.
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
        self, inputs: jax.Array, context: jax.Array | None = None, training: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        if self.output_size is None:
            skip_connection = identity
        else:
            skip_connection = nn.Dense(self.output_size, dtype=self.dtype)
            if self.time_distributed:
                skip_connection = TimeDistributed(skip_connection)

        pre_elu_dense = nn.Dense(
            self.latent_dim,
            dtype=self.dtype,
        )
        dense = nn.Dense(self.latent_dim, dtype=self.dtype)

        if self.time_distributed:
            pre_elu_dense = TimeDistributed(pre_elu_dense)
            dense = TimeDistributed(dense)

        x_skip = skip_connection(inputs)
        x = pre_elu_dense(inputs)

        if context is not None:
            context_dense = nn.Dense(self.latent_dim, dtype=self.dtype)
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
        )(x, training=training)
        x = nn.LayerNorm(dtype=self.dtype)(x + x_skip)
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
    """

    static_categories_sizes: Sequence[int]
    known_categories_sizes: Sequence[int]
    input_static_idx: Sequence[int]
    input_known_real_idx: Sequence[int]
    input_known_categorical_idx: Sequence[int]
    input_observed_idx: Sequence[int]
    latent_dim: int

    num_unknown_inputs: int = 0
    dtype: ComputeDtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> EmbeddingStruct:
        static = jnp.take(inputs, jnp.asarray(self.input_static_idx), axis=-1).astype(jnp.int32)

        if len(self.input_known_real_idx) > 0:
            known_real = jnp.take(inputs, jnp.asarray(self.input_known_real_idx), axis=-1)
        else:
            known_real = None

        if len(self.input_known_categorical_idx) > 0:
            known_categorical = jnp.take(
                inputs, jnp.asarray(self.input_known_categorical_idx), axis=-1
            ).astype(jnp.int32)
        else:
            known_categorical = None

        if len(self.input_observed_idx) > 0:
            observed = jnp.take(inputs, jnp.asarray(self.input_observed_idx), axis=-1)
        else:
            observed = None

        static_input_embeddings = []
        known_real_inputs_embeddings = []

        for i, size in enumerate(self.static_categories_sizes):
            # TODO: use jax.sparse here
            # Static are not time-varying, so we just take 1st time-step.
            embeds_i = nn.Embed(size, self.latent_dim, dtype=self.dtype)(static[:, 0, i])
            static_input_embeddings.append(embeds_i)

        static_input_embeddings = jnp.stack(static_input_embeddings, axis=1)
        if known_real is not None:
            for i in self.input_known_real_idx:
                embeds_i = nn.Dense(self.latent_dim, dtype=self.dtype)(
                    known_real[..., i, jnp.newaxis]
                )
                known_real_inputs_embeddings.append(embeds_i)

        if observed is not None:
            observed_input_embeddings = []
            for i in self.input_observed_idx:
                embeds_i = nn.Dense(self.latent_dim, dtype=self.dtype)(
                    observed[..., i, jnp.newaxis]
                )
                observed_input_embeddings.append(embeds_i)
            observed_input_embeddings = jnp.stack(observed_input_embeddings, axis=-1)
        else:
            observed_input_embeddings = None

        if known_categorical is not None:
            known_categorical_inputs_embeddings = []
            for i, size in enumerate(self.known_categories_sizes):
                embeds_i = nn.Embed(size, self.latent_dim, dtype=self.dtype)(
                    known_categorical[..., i]
                )
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

        return EmbeddingStruct(
            known=known_inputs_embeddings,
            observed=observed_input_embeddings,
            static=static_input_embeddings,
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
    def __call__(self, inputs: jax.Array, training: bool) -> StaticContextStruct:
        flat_x = flatten(inputs)

        mlp_outputs, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            output_size=self.num_static_inputs,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
        )(flat_x, training=training)

        sparse_weights = nn.softmax(mlp_outputs)[..., jnp.newaxis]

        transformed_embeddings = []

        for i in range(self.num_static_inputs):
            embeds_i, _ = GatedResidualNetwork(
                latent_dim=self.latent_dim,
                dropout_rate=self.dropout_rate,
                time_distributed=False,
                dtype=self.dtype,
            )(inputs[:, i][:, jnp.newaxis], training=training)
            transformed_embeddings.append(embeds_i)

        transformed_embeddings = jnp.concatenate(transformed_embeddings, axis=1)
        static_context_vector = jnp.sum(sparse_weights * transformed_embeddings, axis=1)
        context_variable_selection, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
        )(static_context_vector, training=training)
        context_enrichment, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
        )(static_context_vector, training=training)
        context_state_h, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
        )(static_context_vector, training=training)
        context_state_c, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
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
        self, inputs: jax.Array, context: jax.Array, training: bool
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        mlp_outputs, static_gate = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            output_size=self.num_inputs,
            time_distributed=True,
            dtype=self.dtype,
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
    dtype: ComputeDtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array, training: bool) -> jax.Array:
        mask = make_causal_attention_mask(inputs, dtype=self.dtype)
        x = nn.SelfAttention(num_heads=self.num_attention_heads, dtype=self.dtype, use_bias=False)(
            inputs, mask, True
        )
        x, _ = GatedLinearUnit(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
        )(x, training=training)
        x = nn.LayerNorm(dtype=self.dtype)(x + inputs)
        # Nonlinear processing on outputs
        decoded, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
        )(x, training=training)
        # Final skip connection
        decoded, _ = GatedLinearUnit(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
        )(decoded, training=training)
        return decoded


# -------------------------------------------------------------------------------------------------------------


@struct.dataclass
class EmbeddingStruct:
    # all shape = (batch, time, features)
    static: jax.Array
    known: jax.Array
    observed: jax.Array | None


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

    enrichment: jax.Array
    state_h: jax.Array
    state_c: jax.Array
    vector: jax.Array
    weight: jax.Array


# -------------------------------------------------------------------------------------------------------------


@jax.jit
def identity(arr: jax.Array) -> jax.Array:
    return jnp.array(arr)


@jax.jit
def flatten(arr: jax.Array) -> jax.Array:
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


@functools.partial(jax.jit, static_argnames=["dtype"])
def make_causal_attention_mask(x: jax.Array, dtype: DTypeLike = jnp.float32) -> jax.Array:
    """Creates a causal attention mask.

    Args:
      x: The SelfAttention input.
      dtype:

    Returns:
      A tensor of shape `[seq_len, seq_len]`, where each entry is 1 if the
      corresponding positions are causally related and 0 otherwise.
    """
    seq_len = x.shape[1]
    attention_mask = jnp.tril(jnp.ones([seq_len, seq_len], dtype))
    return jnp.expand_dims(attention_mask, 0)
