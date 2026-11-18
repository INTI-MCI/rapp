import pytest  # noqa

from rapp.rotary_stage import RotaryStage, RotaryStageError, RotaryStageMock
from rapp.motion_controller import ESP301, ESP301Error

from rapp.mocks import SerialMock


class ESPMock(ESP301):
    def set_position(*args, **kwargs):
        raise ESP301Error("Error")


def test_rotary_stage(motion_controller):
    _ = RotaryStage(motion_controller)


def test_errors():
    stage = RotaryStage(ESPMock(SerialMock()))

    with pytest.raises(RotaryStageError):
        next(stage)


def test_build_returns_a_mock_only_when_requested(motion_controller):
    mock_stage = RotaryStage.build(
        mock=True, motion_controller=motion_controller, cycles=1, step=90)
    real_stage = RotaryStage.build(
        mock=False, motion_controller=motion_controller, cycles=1, step=90)

    assert isinstance(mock_stage, RotaryStageMock)
    assert not isinstance(real_stage, RotaryStageMock)


def test_current_position(motion_controller):
    stage = RotaryStage(motion_controller, cycles=1, step=90, name="Analyzer")

    assert stage.current_position() == motion_controller.get_position(axis=1)
    assert stage.current_position_for_filename() == stage.current_position()

    stage._index = 1
    assert stage.current_position() == stage._positions[0]

    stage._index = len(stage._positions)
    assert stage.current_position() == stage._positions[-1]

    stage._index = len(stage._positions) + 1
    with pytest.raises(ValueError, match="Index out of range"):
        stage.current_position()


def test_info_current_position(motion_controller):
    stage = RotaryStage(motion_controller, cycles=1, step=90, name="Analyzer")
    stage._index = 1

    assert stage.info_current_position() == "Analyzer angle: {:.2f}°, ".format(stage._positions[0])


def test_rotary_stage_mock(motion_controller):
    stage = RotaryStageMock(motion_controller=motion_controller, cycles=1, step=90, name="Mock")

    initial_position = motion_controller.get_position(axis=1)

    assert stage.current_position_for_filename() is None
    assert list(stage) == [initial_position]
    assert list(stage) == []

    stage.reset()
    assert next(stage) == initial_position

    stage.motor_on()
    stage.set_home(0)
