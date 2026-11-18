import json
import os
from pathlib import Path

import pytest

from rapp import polarimeter
from rapp.adc import ADC, ADCError
from rapp.data_file import DataFile
from rapp.mocks import SerialMock
from rapp.motion_controller import ESP301, ESP301Error
from rapp.rotary_stage import RotaryStage

FILE_DELIMITER = "\t"

CORRECTION_PARAMETERS = {
    "correction_parameters_sensor_0": {"bias": 1.0, "A": 2.0, "b": 1.0},
    "correction_parameters_sensor_1": {"bias": 0.5, "A": 4.0, "b": 2.0},
}


class TemperatureADC:
    def request_temperature(self, channel, flush=True):
        return True

    def read_temperature(self, channel):
        return [20.0 + channel], False


class RecordingDataFile:
    def __init__(self):
        self.rows = []

    def add_row(self, row):
        self.rows.append(row)


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
        temp_wait=0,
        work_dir=tmp_path,
    )

    polarimeter.run(**config)


def test_main_with_pm100_connection_failure(tmp_path, monkeypatch):
    from rapp.pm100 import PM100Error

    def build_raises(*args, **kwargs):
        raise PM100Error("device not found")

    monkeypatch.setattr(polarimeter.PM100, "build", build_raises)

    config = dict(
        cycles=0,
        samples=1,
        delay_position=0,
        hwp_delay_position=0,
        mock_esp=True,
        mock_adc=True,
        enable_pm100=True,
        overwrite=True,
        work_dir=tmp_path,
    )

    polarimeter.run(**config)


def test_main_with_hwp_enabled(tmp_path):
    config = dict(
        cycles=0,
        samples=1,
        delay_position=0,
        hwp_delay_position=0,
        mock_esp=True,
        mock_adc=True,
        overwrite=True,
        hwp_enabled=True,
        delay=0.01,
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
    adc = ADC(SerialMock(), ch0=True, ch1=True, adc_used="ADS_1115")

    motion_controller = ESP301(SerialMock())

    analyzer = RotaryStage(
        motion_controller_mock, cycles, step, delay_position, velocity, axis=1
    )

    hwp = RotaryStage(motion_controller, hwp_cycles, hwp_step, delay_position=hwp_delay, axis=2)

    output_dir = str(Path(str(tmp_path)).joinpath('data'))

    data_file = DataFile(overwrite=True, delimiter=FILE_DELIMITER, output_dir=output_dir)

    temp_file = DataFile(overwrite=True, delimiter=FILE_DELIMITER, output_dir=output_dir)
    qp_temp_file = DataFile(overwrite=True, delimiter=FILE_DELIMITER, output_dir=output_dir)

    p = polarimeter.Polarimeter(adc, analyzer, hwp, data_file, temp_file, qp_temp_file, wait=0)

    failures = p.start(samples, reps=reps)
    assert failures == 1

    p.close()

    files_in_dir = os.listdir(output_dir)
    assert len(files_in_dir) == 8


class NormalizationDetectorMock:
    def start_measurement(self):
        pass

    def fetch_measurement(self):
        return 0.5


def test_read_samples_with_normalization_detector():
    class AdcMock:
        def acquire(self, samples, flush=True):
            return [(1.0, 2.0)] * samples

    p = polarimeter.Polarimeter.__new__(polarimeter.Polarimeter)
    p._adc = AdcMock()
    p._norm_det = NormalizationDetectorMock()

    data = list(p.read_samples(1))

    assert data == [[(1.0, 2.0, 0.5)]]


def test_build_data_filename_includes_hwp_position():
    p = polarimeter.Polarimeter(None, None, None, None)

    assert p._build_data_filename(rep=1, hwp_position=None) == "rep1.csv"
    assert p._build_data_filename(rep=1, hwp_position=40.0) == "hwp40.00-rep1.csv"


def write_correction_parameters(tmp_path):
    filepath = tmp_path / "correction-parameters.json"
    filepath.write_text(json.dumps(CORRECTION_PARAMETERS))
    return filepath


def test_temperature_bias_correction(tmp_path):
    filepath = write_correction_parameters(tmp_path)
    p = polarimeter.Polarimeter(None, None, None, None)

    assert p.temperature_bias_correction(0, filepath=str(filepath), temperature=[10.0]) \
        == pytest.approx(9.0)
    assert p.temperature_bias_correction(1, filepath=str(filepath), temperature=[10.0]) \
        == pytest.approx(9.5)


def test_temperature_linear_correction(tmp_path):
    filepath = write_correction_parameters(tmp_path)
    p = polarimeter.Polarimeter(None, None, None, None)

    assert p.temperature_linear_correction(0, filepath=str(filepath), temperature=[5.0]) \
        == pytest.approx(2.0)
    assert p.temperature_linear_correction(1, filepath=str(filepath), temperature=[10.0]) \
        == pytest.approx(2.0)


def test_request_and_read_temperature_writes_to_matching_file(tmp_path, monkeypatch):
    monkeypatch.setattr(polarimeter, "TEMP_CORRECTION_FILE", str(write_correction_parameters(tmp_path)))

    temperature_file = RecordingDataFile()
    qp_temperature_file = RecordingDataFile()
    p = polarimeter.Polarimeter(
        TemperatureADC(), None, None, None,
        temperature_file=temperature_file, qp_temperature_file=qp_temperature_file)

    parameters = {"position": 0, "hwp_position": 40.0, "rep": 1, "temp_correction": "linear"}
    parameters_req_temperature = {}
    temperature_requested = [False, False]

    p.request_temperature(
        channel=0, parameters_req_temperature=parameters_req_temperature,
        parameters=parameters, temperature_requested=temperature_requested)
    p.read_temperature(
        parameters_req_temperature=parameters_req_temperature,
        temperature_requested=temperature_requested, write=True, channel=0)

    assert len(temperature_file.rows) == 1
    assert len(qp_temperature_file.rows) == 0
    assert temperature_requested[0] is False

    p.request_temperature(
        channel=1, parameters_req_temperature=parameters_req_temperature,
        parameters=parameters, temperature_requested=temperature_requested)
    p.read_temperature(
        parameters_req_temperature=parameters_req_temperature,
        temperature_requested=temperature_requested, write=True, channel=1)

    assert len(temperature_file.rows) == 1
    assert len(qp_temperature_file.rows) == 1

    # Nothing is read when no temperature was requested for that channel.
    p.read_temperature(
        parameters_req_temperature=parameters_req_temperature,
        temperature_requested=temperature_requested, write=True, channel=0)

    assert len(temperature_file.rows) == 1


def test_request_and_read_temperature_with_bias_correction(tmp_path, monkeypatch):
    monkeypatch.setattr(polarimeter, "TEMP_CORRECTION_FILE", str(write_correction_parameters(tmp_path)))

    temperature_file = RecordingDataFile()
    qp_temperature_file = RecordingDataFile()
    p = polarimeter.Polarimeter(
        TemperatureADC(), None, None, None,
        temperature_file=temperature_file, qp_temperature_file=qp_temperature_file)

    parameters = {"position": 0, "hwp_position": 0.0, "rep": 1, "temp_correction": "bias"}
    parameters_req_temperature = {}
    temperature_requested = [False, False]

    p.request_temperature(
        channel=0, parameters_req_temperature=parameters_req_temperature,
        parameters=parameters, temperature_requested=temperature_requested)
    p.read_temperature(
        parameters_req_temperature=parameters_req_temperature,
        temperature_requested=temperature_requested, write=False, channel=0)

    assert len(temperature_file.rows) == 0


if __name__ == "__main__":
    pytest.main()
