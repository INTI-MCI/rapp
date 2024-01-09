from rapp.rotary_stage import RotaryStage

import pytest  # noqa


def test_data_file(motion_controller):
    _ = RotaryStage(motion_controller, init_position=0)
