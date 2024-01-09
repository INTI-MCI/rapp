from rapp.data_file import DataFile

import pytest  # noqa


def test_data_file(monkeypatch, tmp_path):
    data_file = DataFile(output_dir=tmp_path, overwrite=False)
    assert data_file.path is None

    data_file.open('test')
    data_file.write("TEST", new_line=False)
    data_file.write("\n", new_line=False)
    data_file.add_row((1, 2))

    data_file.close()

    monkeypatch.setattr('builtins.input', lambda _: "n")

    data_file.open('test')
    data_file.add_row((3, 4))
    data_file.close()

    monkeypatch.setattr('builtins.input', lambda _: "y")

    data_file.open('test')
    data_file.add_row((5, 6))
    data_file.close()
