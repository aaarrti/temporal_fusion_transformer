from temporal_fusion_transformer.src.experiments.util import (
    serialize_preprocessor,
    deserialize_preprocessor,
)
import jax
import numpy as np
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler


def test_serialize_preprocessor(tmp_path: Path):
    x = jax.random.normal(jax.random.PRNGKey(33), (120, 4))
    x = np.asarray(x)
    labels = [str(i) for i in range(12)]

    sc = StandardScaler()
    le = LabelEncoder()

    x_transformed = sc.fit_transform(x)
    label_transformed = le.fit_transform(labels)

    preprocessor = {"target": sc, "categorical": le}

    serialize_preprocessor(preprocessor, tmp_path.as_posix())

    reloaded_preprocessor = deserialize_preprocessor(tmp_path.as_posix())

    x_transformed_2 = reloaded_preprocessor["target"].transform(x)
    label_transformed_2 = reloaded_preprocessor["categorical"].transform(labels)

    np.testing.assert_allclose(x_transformed, x_transformed_2)
    np.testing.assert_equal(label_transformed, label_transformed_2)
