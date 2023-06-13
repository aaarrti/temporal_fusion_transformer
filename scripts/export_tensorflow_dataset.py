import json

from absl import flags, app, logging

from temporal_fusion_transformer.experiments import ElectricityExperiment
from temporal_fusion_transformer.utils import export_sharded_dataset, map_dict
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "dataset", enum_values=["electricity"], default="electricity", help=None
)
flags.DEFINE_integer("shard_size", default=100_000, help=None)
flags.DEFINE_string("data_path", default="../data", help=None)


def main(_):
    ds, scalers = ElectricityExperiment.from_raw_csv("data/electricity/LD2011_2014.txt")
    ids = (
        set(ds.train["identifier"].reshape(-1))
        .union(set(ds.validation["identifier"].reshape(-1)))
        .union(set(ds.test["identifier"].reshape(-1)))
    )
    metadata = {
        "number_training_samples": len(ds.train["identifier"]),
        "number_validation_samples": len(ds.validation["identifier"]),
        "number_test_samples": len(ds.test["identifier"]),
        "ids": list(ids),
        "scalers": {
            "real": list(scalers.real.keys()),
            "categorical": list(scalers.categorical.keys()),
            "target": list(scalers.target.keys()),
        },
    }
    with open("data/electricity/scalers.pickle", "wb+") as file:
        pickle.dump(scalers, file, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Metadata = {json.dumps(metadata, indent=4)}")
    with open("data/electricity/metadata.json", "w+") as file:
        json.dump(metadata, file)
    logging.info("Exporting sharded train split.")
    export_sharded_dataset(ds.train, "data/electricity/train")
    logging.info("Exporting sharded validation split.")
    export_sharded_dataset(ds.validation, "data/electricity/validation")
    logging.info("Exporting sharded tests split.")
    export_sharded_dataset(ds.test, "data/electricity/test")


if __name__ == "__main__":
    app.run(main)
