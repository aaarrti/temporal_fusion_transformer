from __future__ import annotations

from absl import app, logging, flags
import wget
import os
import shutil
import pyunpack

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "dataset", enum_values=["electricity"], default="electricity", help=None
)
flags.DEFINE_string("data_folder", default="data", help=None)


ELECTRICITY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"


def download_from_url(url: str, output_path: str):
    logging.info("Downloading data from {} to {}".format(url, output_path))
    wget.download(url, output_path)
    logging.info("Done.")


def recreate_folder(path):
    """Deletes and recreates folder."""
    shutil.rmtree(path)
    os.makedirs(path)


def unzip(zip_path: str, output_file: str):
    """Unzips files and checks successful completion."""
    logging.info("Unzipping file: {}".format(zip_path))
    pyunpack.Archive(zip_path).extractall(FLAGS.data_folder)
    # Checks if unzip was successful
    if not os.path.exists(output_file):
        raise ValueError(
            "Error in unzipping process! {} not found.".format(output_file)
        )
    logging.info("Done.")


def download_and_unzip(url: str, zip_path: str, csv_path: str):
    download_from_url(url, zip_path)
    unzip(zip_path, csv_path)


def download_electricity():
    """Downloads electricity dataset from UCI repository."""
    csv_path = os.path.join(FLAGS.data_folder, "LD2011_2014.txt")
    zip_path = csv_path + ".zip"
    download_and_unzip(ELECTRICITY_URL, zip_path, csv_path)
    logging.info("Aggregating to hourly data")


def main(_):
    if FLAGS.dataset == "electricity":
        return download_electricity()


if __name__ == "__main__":
    app.run(main)
