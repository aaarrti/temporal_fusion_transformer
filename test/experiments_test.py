import polars as pl
import tensorflow as tf

# from temporal_fusion_transformer.experiments.electricity import ElectricityExperiment


def test_load_electricity_data():
    df = pl.read_csv("tests/assets/hourly_electricity_mini.csv")
    exp = ElectricityExperiment.from_dataframe(df)

    train_ds = exp.train_split
    validation_ds = exp.validation_split

    assert train_ds.cardinality() == 1
    assert validation_ds.cardinality() == 1

    x_train, y_train = train_ds.as_numpy_iterator().next()
    x_val, y_val = train_ds.as_numpy_iterator()

    assert x_train.shape == ()
    assert y_train.shape == ()
    assert x_val.shape == ()
    assert y_val.shape == ()
