import os

import pytest

from rapp import polarimeter
from rapp.adc import ADCError
from rapp.motion_controller import ESP301, ESP301Error
from rapp.rotary_stage import RotaryStage
from rapp.data_file import DataFile
from rapp.adc import ADC
from rapp.mocks import SerialMock

from pathlib import Path


FILE_DELIMITER = "\t"


def test_main_pm100_mocked(tmp_path):
    config = dict(
        cycles=0,
        samples=1,
        delay_position=0,
        hwp_delay_position=0,
        mock_esp=True,
        mock_adc=True,
        mock_pm100=True,
        overwrite=True,
        work_dir=tmp_path,
    )

    polarimeter.run(**config)


def test_main(tmp_path):
    config = dict(
        cycles=0,
        samples=1,
        delay_position=0,
        hwp_delay_position=0,
        mock_esp=True,
        mock_adc=True,
        mock_pm100=True,
        overwrite=True,
        work_dir=tmp_path,
    )

    polarimeter.run(**config)


def test_failed_adc_acquisition_retries_until_valid_sample():
    class IntermittentADC:
        def __init__(self):
            self.acquisitions = 0
            self.recoveries = 0
            self.results = [None, None, None, [(0.25, 0.5)]]

        def acquire(self, samples, flush=True):
            self.acquisitions += 1
            return self.results.pop(0)

        def recover(self):
            self.recoveries += 1

    p = polarimeter.Polarimeter.__new__(polarimeter.Polarimeter)
    p._adc = IntermittentADC()
    p._norm_det = None

    data = list(p.read_samples(1))

    assert data == [[(0.25, 0.5)]]
    assert p._adc.acquisitions == 4
    assert p._adc.recoveries == 3


def test_failed_adc_acquisition_stops_after_five_attempts():
    class FailedADC:
        def __init__(self):
            self.acquisitions = 0
            self.recoveries = 0

        def acquire(self, samples, flush=True):
            self.acquisitions += 1
            return None

        def recover(self):
            self.recoveries += 1

    p = polarimeter.Polarimeter.__new__(polarimeter.Polarimeter)
    p._adc = FailedADC()
    p._norm_det = None

    with pytest.raises(ADCError, match="failed after 5 attempts"):
        list(p.read_samples(1))

    assert p._adc.acquisitions == 5
    assert p._adc.recoveries == 4


class MotionControllerMock(ESP301):
    i = 0
    should_fail = True

    def set_position(self, position, *args, **kwargs):
        print("iteracion", MotionControllerMock.i)
        if MotionControllerMock.i == 2 and MotionControllerMock.should_fail:
            MotionControllerMock.should_fail = False
            MotionControllerMock.i += 1
            raise ESP301Error

        MotionControllerMock.i += 1

        return position


def test_motion_controller_failure(tmp_path):
    cycles = 1
    step = 90
    samples = 1
    delay_position = 0
    velocity = 4
    hwp_cycles = 0.025
    hwp_step = 9
    hwp_delay = 0
    reps = 3

    motion_controller_mock = MotionControllerMock(SerialMock())
    adc = ADC(SerialMock(), ch0=True, ch1=True)

    motion_controller = ESP301(SerialMock())

    analyzer = RotaryStage(
        motion_controller_mock, cycles, step, delay_position, velocity, axis=1
    )

    hwp = RotaryStage(motion_controller, hwp_cycles, hwp_step, delay_position=hwp_delay, axis=2)

    output_dir = str(Path(str(tmp_path)).joinpath('data'))

    data_file = DataFile(overwrite=True, delimiter=FILE_DELIMITER, output_dir=output_dir)

    temp_file = DataFile(overwrite=True, delimiter=FILE_DELIMITER, output_dir=output_dir)

    p = polarimeter.Polarimeter(adc, analyzer, hwp, data_file, temp_file, wait=0)

    failures = p.start(samples, reps=reps)
    assert failures == 1

    p.close()

    files_in_dir = os.listdir(output_dir)
    assert len(files_in_dir) == 7
