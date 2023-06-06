from absl import flags, app, logging
from temporal_fusion_transformer.experiments import (
    ElectricityExperiment,
    export_sharded_dataset,
)

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "dataset", enum_values=["electricity"], default="electricity", help=None
)
flags.DEFINE_integer("shard_size", default=100_000, help=None)
flags.DEFINE_string("data_path", default="../data", help=None)


def main(_):
    train_ds, val_ds, test_ds = ElectricityExperiment.from_raw_csv(
        "../data/electricity/LD2011_2014.txt"
    )
    logging.info("Exporting sharded train split.")
    export_sharded_dataset(train_ds, "../data/electricity/train")
    logging.info("Exporting sharded validation split.")
    export_sharded_dataset(val_ds, "../data/electricity/validation")
    logging.info("Exporting sharded tests split.")
    export_sharded_dataset(val_ds, "../data/electricity/test")


if __name__ == "__main__":
    app.run(main)
