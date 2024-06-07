from rapp.pm100 import PM100


def test_PM100Mocked():
    PM100.list_resources()

    pm100 = PM100.build("mock")

    assert isinstance(pm100.get_power(), float)

    pm100.start_measurement()
    assert isinstance(pm100.fetch_measurement(), float)

    pm100.set_average_count(100)
    pm100.close()
