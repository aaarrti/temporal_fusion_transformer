from __future__ import annotations

import tensorflow as tf
from absl import flags
from absl_extra import register_task, requires_gpu, run
from keras import mixed_precision
from keras.utils.tf_utils import can_jit_compile

from temporal_fusion_transformer.experiments import (
    DataParams,
    ModelParams,
    OptimizerParams,
    ElectricityExperiment,
)
from temporal_fusion_transformer.modeling import TFTInputs, TemporalFusionTransformer
from temporal_fusion_transformer.train_lib import (
    load_sharded_dataset,
    train_with_fixed_hyper_parameters,
)

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", default=32, help=None)
flags.DEFINE_integer("epochs", default=1, help=None)

NUM_ELECTRICITY_SAMPLES = 1853057


@register_task
@requires_gpu
def main():
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs

    mixed_precision.set_global_policy("mixed_float16")
    element_spec = {
        "identifier": tf.TensorSpec([None, 192, 1], dtype=tf.string),
        "time": tf.TensorSpec([None, 192, 1], dtype=tf.float64),
        "outputs": tf.TensorSpec([None, 24, 1], dtype=tf.float64),
        "inputs_static": tf.TensorSpec([None, 1], dtype=tf.int64),
        "inputs_known_real": tf.TensorSpec([None, 192, 3], dtype=tf.float64),
    }

    def map_fn(arg):
        return (
            TFTInputs(
                static=tf.cast(arg["inputs_static"], tf.int32),
                known_real=tf.cast(arg["inputs_known_real"], tf.float32),
                known_categorical=None,
                observed=None,
            ),
            tf.cast(arg["outputs"], tf.float32),
        )

    train_ds = load_sharded_dataset(
        "../data/electricity/train",
        batch_size,
        element_spec=element_spec,
        map_fn=map_fn,
        drop_remainder=True,
    )

    validation_ds = load_sharded_dataset(
        "../data/electricity/validation",
        batch_size,
        element_spec=element_spec,
        map_fn=map_fn,
        drop_remainder=True,
    )

    hp: ModelParams = ElectricityExperiment.default_params[0]
    op: OptimizerParams = ElectricityExperiment.default_params[1]
    fp: DataParams = ElectricityExperiment.fixed_params

    def make_model():
        return TemporalFusionTransformer(
            static_categories_sizes=fp.static_categories_sizes,
            known_categories_sizes=fp.known_categories_sizes,
            num_encoder_steps=fp.num_encoder_steps,
            hidden_layer_size=hp.hidden_layer_size,
            num_attention_heads=hp.num_attention_heads,
        )

    def make_optimizer():
        return tf.keras.optimizers.Adam(
            learning_rate=op.learning_rate,
            jit_compile=can_jit_compile(True),
            clipnorm=op.max_gradient_norm,
        )

    model, history = train_with_fixed_hyper_parameters(
        make_model,
        make_optimizer,
        train_ds,
        validation_ds,
        epochs=epochs,
        steps_per_epoch=NUM_ELECTRICITY_SAMPLES // batch_size,
    )
    model.save_weights("weights.keras")


if __name__ == "__main__":
    run("temporal_fusion_transformer.scripts.train_model")
