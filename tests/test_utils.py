from rapp.utils import split_number_to_list


def test_split_number_to_list():
    lst = split_number_to_list(1014, size=500)
    assert lst == [500, 500, 14]

    lst = split_number_to_list(400, size=500)
    assert lst == [400]

    lst = split_number_to_list(500, size=250)
    assert lst == [250, 250]
