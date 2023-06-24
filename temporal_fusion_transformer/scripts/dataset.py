from absl import flags
from absl_extra import register_task, run, setup_logging
import temporal_fusion_transformer as tft

setup_logging()
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


@register_task
def _main(_):
    if FLAGS.experiment == "electricity":
        tft.experiments.electricity_experiment.process_raw_data(
            f"{FLAGS.raw_data_path}/electricity/LD2011_2014.txt", FLAGS.save_path
        )
    if FLAGS.experiment == "favorita":
        tft.experiments.favorita_experiment.process_raw_data(
            f"{FLAGS.raw_data_path}/favorita", FLAGS.save_path
        )


def main():
    run("tft_dataset")
