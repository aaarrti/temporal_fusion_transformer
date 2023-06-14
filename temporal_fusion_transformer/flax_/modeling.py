from __future__ import annotations

from typing import Mapping, Tuple, List, Dict

import flax.linen as nn
import jax.nn
import jax.numpy as jnp
import jax

from flax.struct import dataclass
from jaxtyping import Float, Array, Int


@dataclass
class TFTInput:
    static: Int[Array, "batch s"]
    known_real: Float[Array, "batch t k"]
    known_categorical: Int[Array, "batch t c"] | None = None
    observed: Float[Array, "batch t o"] | None = None


@dataclass
class TFTEmbeddings:
    static: Float[Array, "batch s n"]
    real: Float[Array, "batch n r"]
    observed: Float[Array, "batch n o"] | None = None


class TemporalFusionTransformer(nn.Module):
    static_categories_sizes: List[int]
    known_categories_sizes: List[int]
    num_encoder_steps: int
    hidden_layer_size: int
    num_attention_heads: int
    num_quantiles: int
    num_stacks: int = 1
    dropout_rate: float = 0.1
    output_size: int = 1

    @nn.compact
    def __call__(self, inputs: TFTInput, **kwargs) -> Float[Array, "batch t n"]:
        pass


class TFTInputEmbedding(nn.Module):
    static_categories_size: Int[Array, "s"]
    known_categories_size: Int[Array, "rc"]
    hidden_layer_size: int

    @nn.compact
    def __call__(self, inputs: TFTInput, **kwargs) -> TFTEmbeddings:
        static_input_embeddings = []
        known_real_inputs_embeddings = []

        for i, size in enumerate(self.static_categories_size):
            static_input_embeddings.append(
                nn.Embed(size, self.hidden_layer_size)(inputs.static[:, i])
            )

        static_input_embeddings = jnp.stack(static_input_embeddings, axis=1)

        num_known_real_inputs = inputs.known_real.shape[-1]

        for i in range(num_known_real_inputs):
            known_real_inputs_embeddings.append(
                TimeDistributed(nn.Dense)(inputs.known_real[..., i, jnp.newaxis])
            )

        if inputs.observed is not None:
            num_observed_inputs = inputs.observed.shape[-1]
            observed_input_embeddings = []
            for i in range(num_observed_inputs):
                observed_input_embeddings.append(
                    TimeDistributed(nn.Dense(inputs.observed[..., i, jnp.newaxis]))
                )

            observed_input_embeddings = jnp.stack(observed_input_embeddings, axis=-1)
        else:
            observed_input_embeddings = None

        if len(self.known_categories_size) != 0:
            known_categorical_inputs_embeddings = []
            for i, size in enumerate(self.known_categories_size):
                known_categorical_inputs_embeddings.append(
                    nn.Embed(size, self.hidden_layer_size)(
                        inputs.known_categorical[..., i]
                    )
                )
            known_inputs_embeddings = jnp.concat(
                [
                    jnp.stack(known_real_inputs_embeddings, axis=-1),
                    jnp.stack(known_categorical_inputs_embeddings, axis=-1),
                ],
                axis=-1,
            )

        else:
            known_inputs_embeddings = jnp.stack(known_real_inputs_embeddings, axis=-1)

        return TFTEmbeddings(
            static=static_input_embeddings,
            observed=observed_input_embeddings,
            known=known_inputs_embeddings,
        )


@dataclass
class StaticContext:
    enrichment: Float[Array, "batch n"]
    state_h: Float[Array, "batch n"]
    state_c: Float[Array, "batch n"]
    vector: Float[Array, "batch n"]


class StaticCovariatesEncoder(nn.Module):
    hidden_layer_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, inputs: Float[Array, "batch k n"], **kwargs) -> StaticContext:
        pass


class TemporalVariableSelectionNetwork(nn.Module):
    pass


class TemporalFusionDecoder(nn.Module):
    pass


class TimeDistributed(nn.Module):
    pass


class GatedLinearUnit(nn.Module):
    hidden_layer_size: int
    dropout_rate: float
    use_time_distributed: bool

    @nn.compact
    def __call__(
        self,
        inputs: Float[Array, "batch time_steps n"] | Float[Array, "batch n"],
        **kwargs,
    ) -> Tuple[
        Float[Array, "batch time_steps n"] | Float[Array, "batch m"],
        Float[Array, "batch time_steps n"] | Float[Array, "batch m"],
    ]:
        dense = nn.Dense(self.hidden_layer_size)
        pre_activation = nn.Dense(self.hidden_layer_size)
        activation = nn.sigmoid

        if self.use_time_distributed:
            dense = TimeDistributed(dense)
            pre_activation = TimeDistributed(pre_activation)
            activation = TimeDistributed(activation)

        x = nn.Dropout(rate=self.dropout_rate)(inputs)
        x_pre_activation = dense(x)
        x_gated = activation(pre_activation(x))
        x = x_pre_activation * x_gated
        return x, x_gated


# GRN types:
# - use_time_distributed=False, context=None, output_size None - GRN
# - use_time_distributed=True, context None, output_size None - Temporal GRN
# - use_time_distributed=True, context not None, output_size not None - Context Aware GRN ?


@dataclass
class ContextAwareInput:
    inputs: jnp.ndarray
    context: jnp.ndarray


class ContextAwareGRN(nn.Module):
    hidden_layer_size: int
    dropout_rate: float
    output_size: int
    # it is always time-distributed.

    @nn.compact
    def __call__(
        self, inputs: ContextAwareInput, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_skip = TimeDistributed(nn.Dense(self.output_size))(inputs.inputs)
        x = TimeDistributed(nn.Dense(self.hidden_layer_size))(inputs.inputs)
        x = x + TimeDistributed(nn.Dense(self.hidden_layer_size))(inputs.context)

        x = jax.nn.elu(x)
        x = TimeDistributed(nn.Dense(self.hidden_layer_size))(x)
        x, gate = GatedLinearUnit(
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
        )(x)
        x = nn.LayerNorm()(x + x_skip)
        return x, gate


class GatedResidualNetwork(nn.Module):
    hidden_layer_size: int
    dropout_rate: float
    use_time_distributed: bool

    @nn.compact
    def __call__(
        self, inputs: Float[Array, "batch n"], **kwargs
    ) -> Tuple[Float[Array, "batch n"], Float[Array, "batch n"]]:
        x_skip = inputs
        dense_1 = nn.Dense(self.hidden_layer_size)
        dense_2 = nn.Dense(self.hidden_layer_size)
        if self.use_time_distributed:
            dense_1 = TimeDistributed(dense_1)
            dense_2 = TimeDistributed(dense_2)

        x = dense_1(inputs)
        x = jax.nn.elu(x)
        x = dense_2(x)
        x, gate = GatedLinearUnit(
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=self.use_time_distributed,
        )(x)
        x = nn.LayerNorm()(x + x_skip)
        return x, gate


class TemporalEncoderBlock(nn.Module):
    pass


def make_causal_attention_mask(self_attn_inputs):
    pass
