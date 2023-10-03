from absl import flags
from absl_extra.logging_utils import setup_logging
from absl_extra.notifier import NoOpNotifier
from absl_extra.tasks import register_task, run

import temporal_fusion_transformer as tft

setup_logging()
FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "experiment",
    enum_values=["electricity", "favorita"],
    help="Name of the experiment_name.",
    default=None,
    required=True,
)
flags.DEFINE_string("data_dir", default="data", help="Directory into which dataset should be downloaded.")


@register_task(name="parquet", notifier=NoOpNotifier())
def convert_to_parquet():
    experiment_name = FLAGS.experiment
    data_dir = FLAGS.data_dir

    if experiment_name == "electricity":
        experiment = tft.experiments.Electricity()
    elif experiment_name == "favorita":
        experiment = tft.experiments.Favorita()
    else:
        raise RuntimeError(f"Unknown experiment: {experiment_name}")

    experiment.convert_to_parquet(f"{data_dir}/{experiment_name}")


@register_task(name="dataset", notifier=NoOpNotifier())
def make_dataset():
    data_dir, experiment_name = FLAGS.data_dir, FLAGS.experiment
    data_dir = f"{data_dir}/{experiment_name}"

    if experiment_name == "electricity":
        ex = tft.experiments.Electricity()
    elif experiment_name == "favorita":
        ex = tft.experiments.Favorita()
    else:
        raise RuntimeError("this is unexpected")

    ex.make_dataset(data_dir, save_dir=data_dir)


if __name__ == "__main__":
    run()
