from importlib import util

if util.find_spec("polars") is not None and util.find_spec("tensorflow") is not None:
    from temporal_fusion_transformer.src.datasets.air_passengers import (  # noqa
        AirPassengerPreprocessor,
    )
    from temporal_fusion_transformer.src.datasets.base import (  # noqa
        MultiHorizonTimeSeriesDataset,
        PreprocessorBase,
    )
    from temporal_fusion_transformer.src.datasets.electricity import (  # noqa
        ElectricityDataset,
        ElectricityPreprocessor,
    )
    from temporal_fusion_transformer.src.datasets.favorita import (  # noqa
        FavoritaDataset,
        FavoritaPreprocessor,
    )
