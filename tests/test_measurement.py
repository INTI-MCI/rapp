import pandas as pd  # noqa
import pytest

from rapp.measurement import (
    COLUMN_ANGLE,
    COLUMN_CH0,
    COLUMN_CH1,
    Measurement,
    parse_input_parameters_from_filepath,
    process_temperature_data,
)
from rapp.analysis.phase_diff import plot_phase_difference  # noqa


A0_NOISE = [2.6352759502752957e-06, 0.0003747564924374617]
A1_NOISE = [3.817173720425239e-06, 0.0002145422291402638]

MEASUREMENT_CONTENT = "\n".join([
    "# header",
    "ANGLE,CH0,CH1",
    "0.0,1.0,2.0",
    "1.0,1.5,2.5",
])

TEMPERATURE_CONTENT = "\n".join([
    "# Tiempo-espera-60 s",
    "ANGLE,TEMPERATURE,HWP-POS,REP",
    "0.0,20.0,0.0,1",
    "1.0,22.0,0.0,1",
    "0.0,21.0,0.0,2",
    "1.0,23.0,0.0,2",
    "0.0,30.0,10.0,1",
])


def test_phase_difference(tmp_path):
    cycles = 0.5
    samples = 50
    step = 20

    angles = [0, 1, 28, 44, 60, 87]

    for angle in angles:
        print("True phase diff: {}".format(angle))
        measurement = Measurement.simulate(angle, cycles=cycles, samples=samples, step=step)
        xs, s1, s2, s1_sigma, s2_sigma, res = measurement.phase_diff(method='NLS', allow_nan=True)
        plot_phase_difference((xs, s1, s2, s1_sigma, s2_sigma, res), work_dir=tmp_path, show=False)
        assert angle == pytest.approx(res.value, abs=0.001)


def test_from_file_loads_temperature_of_current_rep(tmp_path):
    (tmp_path / "temperature.csv").write_text(TEMPERATURE_CONTENT)
    filepath = tmp_path / "measurement-rep2-cycles1-step10-samples2.csv"
    filepath.write_text(MEASUREMENT_CONTENT)

    measurement = Measurement.from_file(filepath)

    assert measurement._cycles == 1
    assert measurement._step == 10
    assert measurement._samples == 2
    assert list(measurement.temperature) == [21.0, 23.0]
    assert list(measurement.angle_temp) == [0.0, 1.0]
    assert list(measurement.angles) == [0.0, 1.0]


def test_from_file_without_temperature_file_accepts_str_path(tmp_path):
    filepath = tmp_path / "measurement-cycles1-step10-samples2.csv"
    filepath.write_text(MEASUREMENT_CONTENT)

    measurement = Measurement.from_file(str(filepath))

    assert measurement.temperature is None
    assert measurement.angle_temp is None


def test_process_temperature_data_from_folder_and_file(tmp_path):
    (tmp_path / "temperature.csv").write_text(TEMPERATURE_CONTENT)

    mean_temps, max_temps, min_temps = process_temperature_data(
        tmp_path, "temperature", hwp_position=0.0)

    assert list(mean_temps) == [21.0, 22.0]
    assert list(max_temps) == [22.0, 23.0]
    assert list(min_temps) == [20.0, 21.0]

    mean_temps, max_temps, min_temps = process_temperature_data(
        tmp_path / "temperature.csv", "unused", hwp_position=10.0)

    assert list(mean_temps) == [30.0]
    assert list(max_temps) == [30.0]
    assert list(min_temps) == [30.0]


def test_process_temperature_data_missing_file_returns_none(tmp_path):
    assert process_temperature_data(
        tmp_path, "temperature", hwp_position=0.0, raise_error=False) is None


def test_process_temperature_data_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        process_temperature_data(tmp_path, "temperature", hwp_position=0.0)


def test_parse_input_parameters_from_filepath_accepts_str():
    params = parse_input_parameters_from_filepath(
        "/tmp/measurement-cycles2-step10.5-samples3.csv")

    assert params == {"cycles": 2.0, "step": 10.5, "samples": 3}


def test_average_data_with_one_sample_per_angle():
    data = pd.DataFrame({
        COLUMN_ANGLE: [0.0, 1.0],
        COLUMN_CH0: [1.0, 2.0],
        COLUMN_CH1: [3.0, 4.0],
    })
    measurement = Measurement(data)

    xs, s1, s2, s1_sigma, s2_sigma = measurement.average_data()

    assert list(xs) == [0.0, 1.0]
    assert list(s1) == [1.0, 2.0]
    assert list(s2) == [3.0, 4.0]
    assert s1_sigma is None
    assert s2_sigma is None


def test_average_data_with_several_samples_per_angle():
    data = pd.DataFrame({
        COLUMN_ANGLE: [0.0, 0.0, 1.0, 1.0],
        COLUMN_CH0: [1.0, 3.0, 2.0, 4.0],
        COLUMN_CH1: [3.0, 5.0, 4.0, 6.0],
    })

    _, _, _, s1_sigma, s2_sigma = Measurement(data).average_data()

    assert s1_sigma is not None
    assert s2_sigma is not None
