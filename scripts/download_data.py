from absl import app, logging, flags
import wget
import os
import shutil
import pyunpack

# TODO migrate to polars
import pandas as pd
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "dataset", enum_values=["electricity", "favorita"], default="electricity", help=None
)
flags.DEFINE_string("data_folder", default=".", help=None)


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


def download_favorita():
    logging.info("Download favorita dataset.")


def download_electricity():
    """Downloads electricity dataset from UCI repository."""
    csv_path = os.path.join(FLAGS.data_folder, "LD2011_2014.txt")
    zip_path = csv_path + ".zip"

    download_and_unzip(ELECTRICITY_URL, zip_path, csv_path)

    print("Aggregating to hourly data")

    df = pd.read_csv(csv_path, index_col=0, sep=";", decimal=",")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Used to determine the start and end dates of a series
    output = df.resample("1h").mean().replace(0.0, np.nan)

    earliest_time = output.index.min()

    df_list = []
    for label in output:
        print("Processing {}".format(label))
        srs = output[label]

        start_date = min(srs.fillna(method="ffill").dropna().index)
        end_date = max(srs.fillna(method="bfill").dropna().index)

        active_range = (srs.index >= start_date) & (srs.index <= end_date)
        srs = srs[active_range].fillna(0.0)

        tmp = pd.DataFrame({"power_usage": srs})
        date = tmp.index
        tmp["t"] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time
        ).days * 24
        tmp["days_from_start"] = (date - earliest_time).days
        tmp["categorical_id"] = label
        tmp["date"] = date
        tmp["id"] = label
        tmp["hour"] = date.hour
        tmp["day"] = date.day
        tmp["day_of_week"] = date.dayofweek
        tmp["month"] = date.month

        df_list.append(tmp)

    output = pd.concat(df_list, axis=0, join="outer").reset_index(drop=True)

    output["categorical_id"] = output["id"].copy()
    output["hours_from_start"] = output["t"]
    output["categorical_day_of_week"] = output["day_of_week"].copy()
    output["categorical_hour"] = output["hour"].copy()

    # Filter to match range used by other academic papers
    output = output[
        (output["days_from_start"] >= 1096) & (output["days_from_start"] < 1346)
    ].copy()

    output_mini = output[:100]
    output.to_csv(f"{FLAGS.data_folder}/hourly_electricity.csv")
    output_mini.to_csv(f"{FLAGS.data_folder}/hourly_electricity_mini.csv")

    logging.info("Done.")


def main(_):
    if FLAGS.dataset == "electricity":
        return download_electricity()
    if FLAGS.dataset == "favorita":
        return download_favorita()


if __name__ == "__main__":
    app.run(main)
