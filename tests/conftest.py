import pytest


@pytest.fixture(autouse=True)
def disable_capsys(capsys):
    with capsys.disabled():
        yield
