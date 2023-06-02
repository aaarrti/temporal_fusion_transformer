import pytest
import tensorflow as tf
from temporal_fusion_transformer.modeling import (
    TemporalFusionTransformer,
    TFTInputs,
    StaticCovariatesEncoder,
    TFTInputEmbedding,
    TemporalVariableSelectionNetwork,
    ContextAwareInputs,
    TemporalFusionDecoder,
    DecoderInputs,
)


x_batch = TFTInputs(
    static=tf.ones((8, 4), dtype=tf.int32),
    known_real=tf.ones((8, 30, 3), dtype=tf.float32),
    known_categorical=(tf.ones((8, 30, 2), dtype=tf.int32)),
    observed=tf.random.uniform((8, 30, 1), dtype=tf.float32),
)
static_categories_sizes = [2, 2]
known_categories_sizes = [4]
n_time_steps = 30
batch_size = 8
hidden_layer_size = 5

# tf.config.run_functions_eagerly(True)


def test_input_embedding():
    layer = TFTInputEmbedding(
        static_categories_sizes, known_categories_sizes, hidden_layer_size
    )
    x_embeds = layer(x_batch)
    # Same shapes as in the original implementation.
    assert x_embeds.static.shape == (batch_size, 2, hidden_layer_size)
    assert x_embeds.known.shape == (batch_size, n_time_steps, hidden_layer_size, 4)
    assert x_embeds.observed.shape == (batch_size, n_time_steps, hidden_layer_size, 1)


def test_static_covarites_encoder():
    layer = StaticCovariatesEncoder(hidden_layer_size)
    static_context = layer(tf.random.uniform((batch_size, 2, hidden_layer_size)))
    # Same shapes as in the original implementation.
    assert static_context.vector.shape == (batch_size, hidden_layer_size)
    assert static_context.enrichment.shape == (batch_size, hidden_layer_size)
    assert static_context.state_h.shape == (batch_size, hidden_layer_size)
    assert static_context.state_c.shape == (batch_size, hidden_layer_size)


@pytest.mark.parametrize("time_steps, features", [(25, 4), (5, 3)])
def test_temporal_variable_selection_network(time_steps, features):
    layer = TemporalVariableSelectionNetwork(hidden_layer_size)
    x = ContextAwareInputs(
        inputs=tf.random.uniform((batch_size, time_steps, hidden_layer_size, features)),
        context=tf.random.uniform((batch_size, hidden_layer_size)),
    )
    feature, flags, _ = layer(x)
    # Same shapes as in the original implementation.
    assert feature.shape == (batch_size, time_steps, hidden_layer_size)
    assert flags.shape == (batch_size, time_steps, 1, features)


def test_decoder():
    layer = TemporalFusionDecoder(
        4, hidden_layer_size=hidden_layer_size, dropout_rate=0
    )
    decoder_in = DecoderInputs(
        lstm_outputs=tf.random.uniform((batch_size, n_time_steps, hidden_layer_size)),
        input_embeddings=tf.random.uniform(
            (batch_size, n_time_steps, hidden_layer_size)
        ),
        context_vector=tf.random.uniform((batch_size, hidden_layer_size)),
    )
    decoder_out, _ = layer(decoder_in)
    assert decoder_out.shape == (batch_size, n_time_steps, hidden_layer_size)


def test_full_model():
    model = TemporalFusionTransformer(
        num_encoder_steps=25,
        num_attention_heads=4,
        hidden_layer_size=hidden_layer_size,
        static_categories_sizes=static_categories_sizes,
        known_categories_sizes=known_categories_sizes,
        return_attentions=True,
        dropout_rate=0,
        quantiles=[1, 2, 3],
        output_size=1,
    )

    logits, attentions = model(x_batch)
    tf.debugging.check_numerics(logits, "Test Failed.")
    # 3 is default number of quantiles.
    assert logits.shape == (batch_size, 5, 3)
    assert attentions.decoder_self_attn.shape == (
        batch_size,
        4,
        n_time_steps,
        n_time_steps,
    )
    # FIXME
    # assert attentions.future_flags.shape == (batch_size, 5, 4)
    # assert attentions.historical_flags.shape == (batch_size, n_time_steps, 5)
    # assert attentions.static_flags.shape == (batch_size, 1)
