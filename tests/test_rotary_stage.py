from rapp.rotary_stage import RotaryStage, RotaryStageError
from rapp.motion_controller import ESP301, ESP301Error

from rapp.mocks import SerialMock
import pytest  # noqa


class ESPMock(ESP301):
    def set_position(*args, **kwargs):
        raise ESP301Error("Error")


def test_rotary_stage(motion_controller):
    _ = RotaryStage(motion_controller)


def test_errors():
    stage = RotaryStage(ESPMock(SerialMock()))

    with pytest.raises(RotaryStageError):
        next(stage)
