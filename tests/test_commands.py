import pytest

from rapp.commands import phase_diff, plot_raw


def test_phase_diff_command_runs_on_a_single_file(tmp_path, monkeypatch):
    called = {}

    def fake_phase_difference_from_file(filepath, **kwargs):
        called["filepath"] = filepath
        called.update(kwargs)

    monkeypatch.setattr(phase_diff.phase_diff, "phase_difference_from_file",
                        fake_phase_difference_from_file)
    filepath = tmp_path / "measurement.csv"
    filepath.write_text("ANGLE,CH0,CH1\n0.0,1.0,2.0\n")

    phase_diff.run(filepath, method="NLS")

    assert called["filepath"] == filepath
    assert called["method"] == "NLS"


def test_phase_diff_command_runs_on_a_folder(tmp_path, monkeypatch):
    called = {}

    def fake_phase_difference_from_folder(folder, **kwargs):
        called["folder"] = folder
        called.update(kwargs)

    monkeypatch.setattr(phase_diff.phase_diff, "phase_difference_from_folder",
                        fake_phase_difference_from_folder)
    folder = tmp_path / "measurements"
    folder.mkdir()

    phase_diff.run(folder, appended_measurements=2, method="NLS")

    assert called["folder"] == folder
    assert called["appended_measurements"] == 2


def test_phase_diff_command_runs_instantaneous_phase_difference(tmp_path, monkeypatch):
    called = {}

    def fake_instantaneous(filepath, **kwargs):
        called["filepath"] = filepath
        called.update(kwargs)

    monkeypatch.setattr(phase_diff.phase_diff, "instantaneous_phase_difference",
                        fake_instantaneous)
    filepath = tmp_path / "measurement.csv"
    filepath.write_text("ANGLE,CH0,CH1\n0.0,1.0,2.0\n")

    phase_diff.run(filepath, instantaneous=True, norm=True)

    assert called["filepath"] == filepath
    assert called["norm"] is True


def test_phase_diff_command_rejects_a_folder_for_instantaneous_analysis(tmp_path):
    folder = tmp_path / "measurements"
    folder.mkdir()

    with pytest.raises(ValueError, match="Cannot plot instantaneous"):
        phase_diff.run(folder, instantaneous=True)


def test_plot_raw_command_plots_temperature(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(plot_raw.raw, "plot_temperature",
                        lambda **kwargs: called.update(kwargs))

    plot_raw.run(filepath=tmp_path / "temperature.csv", temperature=True)

    assert called["filepath"] == tmp_path / "temperature.csv"


def test_plot_raw_command_plots_raw_data(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(plot_raw.raw, "plot_raw_from_file",
                        lambda **kwargs: called.update(kwargs))

    plot_raw.run(filepath=tmp_path / "measurement.csv", temperature=False)

    assert called["filepath"] == tmp_path / "measurement.csv"
