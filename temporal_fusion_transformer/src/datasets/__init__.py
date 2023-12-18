from importlib import util

if util.find_spec("polars") is not None and util.find_spec("tensorflow") is not None:
    from temporal_fusion_transformer.src.datasets.base import (
        MultiHorizonTimeSeriesDataset,
        PreprocessorBase,
    )  # noqa
    from temporal_fusion_transformer.src.datasets.air_passengers import (
        AirPassengersDataset,
        AirPassengerPreprocessor,
    )  # noqa
    from temporal_fusion_transformer.src.datasets.electricity import (
        ElectricityDataset,
        ElectricityPreprocessor,
    )  # noqa
    from temporal_fusion_transformer.src.datasets.favorita import (
        FavoritaDataset,
        FavoritaPreprocessor,
    )  # noqa
