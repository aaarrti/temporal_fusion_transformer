import json
import numpy as np
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
        "data/electricity/LD2011_2014.txt"
    )
    metadata = {
        "number_training_samples": len(train_ds["identifier"]),
        "number_validation_samples": len(val_ds["identifier"]),
        "number_test_samples": len(test_ds["identifier"]),
    }
    logging.info(f"Metadata = {json.dumps(metadata, indent=4)}")
    with open("data/electricity/metadata.json", "w+") as f:
        json.dump(metadata, f)
    logging.info("Exporting sharded train split.")
    np.savez("data/electricity/train.npz", **train_ds)
    export_sharded_dataset(train_ds, "data/electricity/train")

    logging.info("Exporting sharded validation split.")
    np.savez("data/electricity/validation.npz", **val_ds)
    export_sharded_dataset(val_ds, "data/electricity/validation")

    logging.info("Exporting sharded tests split.")
    export_sharded_dataset(val_ds, "data/electricity/test")
    np.savez("data/electricity/test.npz", **test_ds)


if __name__ == "__main__":
    app.run(main)
