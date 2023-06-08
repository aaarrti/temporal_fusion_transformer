from __future__ import annotations

import tensorflow as tf
from absl import flags
from absl_extra import register_task, requires_gpu, run, make_strategy
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
flags.DEFINE_enum(
    "precision",
    default="float32",
    enum_values=[
        "float32",
        "mixed_float16",
        "float16",
        "bfloat32",
        "mixed_bfloat16",
        "bfloat16",
    ],
    help=None,
)

NUM_ELECTRICITY_SAMPLES = 1853057
# tf.config.run_functions_eagerly(True)
tf.config.set_soft_device_placement(True)


@register_task
@requires_gpu
def main():
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs

    steps_per_epoch = NUM_ELECTRICITY_SAMPLES // batch_size
    mixed_precision.set_global_policy(FLAGS.precision)
    if can_jit_compile():
        tf.config.optimizer.set_jit("autoclustering")

    element_spec = {
        "identifier": tf.TensorSpec([None, 192, 1], dtype=tf.string),
        "time": tf.TensorSpec([None, 192, 1], dtype=tf.float32),
        "outputs": tf.TensorSpec([None, 24, 1], dtype=tf.float32),
        "inputs_static": tf.TensorSpec([None, 1], dtype=tf.int32),
        "inputs_known_real": tf.TensorSpec([None, 192, 3], dtype=tf.float32),
    }

    def map_fn(arg):
        return (
            TFTInputs(
                static=tf.cast(arg["inputs_static"], tf.int32),
                known_real=tf.cast(
                    arg["inputs_known_real"],
                    mixed_precision.global_policy().compute_dtype,
                ),
                known_categorical=None,
                observed=None,
            ),
            tf.cast(arg["outputs"], mixed_precision.global_policy().compute_dtype),
        )

    train_ds = load_sharded_dataset(
        "data/electricity/train",
        batch_size,
        element_spec=element_spec,
        map_fn=map_fn,
        drop_remainder=True,
    ).repeat(epochs)

    validation_ds = load_sharded_dataset(
        "data/electricity/validation",
        batch_size,
        element_spec=element_spec,
        map_fn=map_fn,
        drop_remainder=True,
    ).repeat(epochs)

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
            dtype=mixed_precision.global_policy().variable_dtype,
            unroll_lstm=True,
        )

    def make_optimizer():
        return tf.keras.optimizers.Adam(
            learning_rate=tf.keras.experimental.CosineDecay(
                op.learning_rate, decay_steps=steps_per_epoch * epochs, alpha=0.05
            ),
            jit_compile=can_jit_compile(True),
            clipnorm=op.max_gradient_norm,
        )

    model, history = train_with_fixed_hyper_parameters(
        make_model,
        make_optimizer,
        train_ds,
        validation_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )
    model.save_weights("weights.keras")


if __name__ == "__main__":
    run("temporal_fusion_transformer.scripts.train_model")
