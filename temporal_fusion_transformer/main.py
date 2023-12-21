from __future__ import annotations

import gc
import logging.config

logging.basicConfig(
    format="%(asctime)s:%(name)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s",
    level=logging.DEBUG,
)
import os  # noqa: E402

os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.utils import set_random_seed  # noqa: E402
from keras.utils import plot_model

set_random_seed(33)
import argparse  # noqa: E402
import logging  # noqa: E402

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import mixed_precision  # noqa

import temporal_fusion_transformer as tft  # noqa: E402
from temporal_fusion_transformer.src.datasets.utils import persist_dataset  # noqa: E402

# tf.debugging.enable_check_numerics()
# tf.debugging.experimental.enable_dump_debug_info("logs/", "FULL_HEALTH", circular_buffer_size=-1)
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
# tf.debugging.set_log_device_placement(True)

log = logging.getLogger(__name__)
# mixed_precision.set_global_policy("mixed_float16")


def train_model_task(data_dir: str, config: tft.Config):
    dataset = tft.load_dataset_from_config(data_dir, config)

    tag = tft.utils.make_timestamp_tag()

    model = tft.train_model_from_config(
        dataset=dataset,
        config=config,
        callbacks=[
            tft.TerminateOnNaN(),
            keras.callbacks.TensorBoard(log_dir=f"{data_dir}/{tag}/tensorboard"),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
                start_from_epoch=10,
            ),
        ],
    )
    # model.save_weights(f"{data_dir}/{tag}/model.weights.h5")


def create_dataset_task(data_dir: str, experiment: str, config: tft.Config):
    dataset: tft.MultiHorizonTimeSeriesDataset = {
        "electricity": tft.datasets.ElectricityDataset,
        "favorita": tft.datasets.FavoritaDataset,
        "air_passengers": tft.datasets.AirPassengersDataset,
    }[experiment](config)

    dataset.convert_to_parquet(data_dir)

    train_ds, val_ds, test_df, preprocessor = dataset.make_dataset(data_dir)
    gc.collect()
    persist_dataset(
        train_ds,
        val_ds,
        test_df,
        test_split_save_format=config.test_split_save_format,
        compression=config.compression,
        save_dir=data_dir,
    )
    preprocessor.save(data_dir)


def optimizer_hyperparams_task(data_dir: str, config: tft.Config):
    tag = tft.utils.make_timestamp_tag()
    dataset = tft.load_dataset_from_config(data_dir, config)
    tft.optimizer_hyperparameters(config, dataset, logdir=f"{data_dir}/{tag}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["dataset", "model", "hyperparams"])
    parser.add_argument("dataset", choices=["electricity", "favorita", "air_passengers"])

    parser.add_argument("--data-dir", default="data")
    # parser.add_argument(
    #    "--mixed-precision", type=bool, default=False, action=argparse.BooleanOptionalAction
    # )

    args = parser.parse_args()

    log.info(f"TF devices = {tf.config.get_visible_devices()}")
    kwargs = {
        "data_dir": f"{args.data_dir}/{args.dataset}",
        "config": tft.Config.read_from_file(
            f"temporal_fusion_transformer/configs/{args.dataset}.toml"
        ),
    }

    match args.task:
        case "dataset":
            create_dataset_task(**kwargs, experiment=args.dataset)
        case "model":
            train_model_task(**kwargs)
        case "hyperparams":
            optimizer_hyperparams_task(**kwargs)
