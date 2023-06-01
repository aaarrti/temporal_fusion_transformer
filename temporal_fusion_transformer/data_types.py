from enum import IntEnum, auto


class DataTypes(IntEnum):
    """Defines numerical types of each column."""

    REAL_VALUED = auto()
    CATEGORICAL = auto()
    DATE = auto()


class InputTypes(IntEnum):
    """Defines input types of each column."""

    TARGET = auto()
    OBSERVED_INPUT = auto()
    KNOWN_INPUT = auto()
    STATIC_INPUT = auto()
    # Single column used as an entity identifier
    ID = auto()
    # Single column exclusively used as a time index
    TIME = auto()
