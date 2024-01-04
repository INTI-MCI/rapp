import pytest

from rapp import mocks
from rapp.motion_controller import ESP301


@pytest.fixture
def motion_controller():
    return ESP301(mocks.SerialMock())
