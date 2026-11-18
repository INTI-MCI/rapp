from pathlib import Path

from rapp.utils import (
    sort_files_by_hwp_and_rep,
    sort_files_by_rep,
    split_number_to_list,
)


def test_split_number_to_list():
    lst = split_number_to_list(1014, size=500)
    assert lst == [500, 500, 14]

    lst = split_number_to_list(400, size=500)
    assert lst == [400]

    lst = split_number_to_list(500, size=250)
    assert lst == [250, 250]


def test_sort_files_by_rep():
    files = ["rep10.csv", "rep2.csv", "rep1.csv"]

    assert sort_files_by_rep(files) == ["rep1.csv", "rep2.csv", "rep10.csv"]


def test_sort_files_by_hwp_and_rep():
    files = [
        Path("hwp0.00-rep1.csv"),
        Path("hwp0.00-rep2.csv"),
        Path("hwp10.00-rep1.csv"),
        Path("hwp10.00-rep2.csv"),
    ]

    sorted_files, hwp_positions, max_rep = sort_files_by_hwp_and_rep(files)

    assert hwp_positions == [0.0, 10.0]
    assert max_rep == 2
    assert sorted_files.shape == (2, 2)
    assert sorted_files[0, 0].name == "hwp0.00-rep1.csv"
    assert sorted_files[0, 1].name == "hwp0.00-rep2.csv"
    assert sorted_files[1, 0].name == "hwp10.00-rep1.csv"
    assert sorted_files[1, 1].name == "hwp10.00-rep2.csv"


def test_sort_files_by_hwp_and_rep_ignores_files_without_rep():
    files = [
        Path("hwp0.00-rep1.csv"),
        Path("notes.csv"),
    ]

    sorted_files, hwp_positions, max_rep = sort_files_by_hwp_and_rep(files)

    assert hwp_positions == [0.0]
    assert max_rep == 1
    assert sorted_files[0, 0].name == "hwp0.00-rep1.csv"
