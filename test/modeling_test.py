import tensorflow as tf
from temporal_fusion_transformer.modeling import TemporalFusionTransformer, TFTInputs


# All inputs (None, 192, 5)

tf.config.run_functions_eagerly(True)


def test_tft_forward_pass():
    inputs = TFTInputs(
        static=(tf.ones((8,), dtype=tf.int32), tf.zeros((8,), dtype=tf.int32)),
        known_real=(tf.random.uniform((8, 30)),),
        known_categorical=(tf.ones((8, 30), dtype=tf.int32) * 3,),
        observed=(tf.random.uniform((8, 30)),),
    )

    model = TemporalFusionTransformer(
        num_static_inputs=2,
        num_known_real_inputs=1,
        num_known_categorical_inputs=1,
        num_time_steps=30,
        num_encoder_steps=25,
        num_attention_heads=4,
        num_observed_inputs=1,
        hidden_layer_size=5,
        static_categories_sizes=[2, 2],
        known_categories_sizes=[4],
    )
    model.compile(run_eagerly=True)

    logits = model(inputs).logits
    tf.debugging.check_numerics(logits, "Test Failed.")
    assert logits.shape == ()
