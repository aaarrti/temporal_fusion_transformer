from absl import flags, app, logging

from temporal_fusion_transformer.src.experiments import (
    electricity_experiment,
    favorita_experiment,
)

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "experiment",
    enum_values=["electricity", "favorita"],
    required=True,
    help="Name of the experiment",
    default=None,
)
flags.DEFINE_string(
    "save_path", default="datasets", help="Path where to persist the dataset"
)
flags.DEFINE_string("raw_data_path", default="raw_data", help="Path to raw data.")


def main(_):
    if FLAGS.experiment == "electricity":
        electricity_experiment.process_raw_data(
            f"{FLAGS.raw_data_path}/electricity/LD2011_2014.txt", FLAGS.save_path
        )
    if FLAGS.experiment == "favorita":
        favorita_experiment.process_raw_data(
            f"{FLAGS.raw_data_path}/favorita", FLAGS.save_path
        )


if __name__ == "__main__":
    app.run(main)
