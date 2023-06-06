from absl import flags, app, logging
from temporal_fusion_transformer.experiments import ElectricityExperiment, export_data_as_tensorflow_dataset

FLAGS = flags.FLAGS
flags.DEFINE_enum("dataset", enum_values=["electricity"], default="electricity", help=None)
flags.DEFINE_integer("shard_size", default=100_000, help=None)
flags.DEFINE_string("data_path", default="../data", help=None)


def main(_):
    logging.info(f"Loading {FLAGS.dataset} dataset.")
    train_ds, val_ds = ElectricityExperiment.from_raw_csv("../data/electricity/LD2011_2014.txt")
    logging.info(f"Exporting {FLAGS.dataset} dataset.")
    export_data_as_tensorflow_dataset(train_ds, "../data/electricity/train")
    export_data_as_tensorflow_dataset(val_ds, "../data/electricity/validation")


if __name__ == "__main__":
    app.run(main)