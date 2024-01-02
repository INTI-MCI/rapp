from rapp import esp
from rapp import mocks

import pytest  # noqa
import serial


@pytest.fixture
def rotator_mock():
    return esp.ESP(mocks.SerialMock())


def test_esp_init():
    with pytest.raises(esp.ESPError):
        esp.ESP()

    esp.ESP(mocks.SerialMock())
    rotator = esp.ESP(mocks.SerialMock(), useaxes=[1, 2])

    assert rotator.axes_in_use == [1, 2]

    with pytest.raises(esp.ESPError):
        rotator = esp.ESP(mocks.SerialMock(), reset=True, useaxes=[1, 2])

    with pytest.raises(esp.ESPError):
        rotator = esp.ESP(mocks.SerialMock(), initpos=4)


def test_build(monkeypatch):
    with pytest.raises(esp.ESPError):
        esp.ESP.build(dev='BADDEVICE')

    monkeypatch.setattr(serial, "Serial", lambda *x, **y: mocks.SerialMock())
    esp.ESP.build()


def test_esp(rotator_mock):
    rotator_mock.motor_off(axis=2)

    with pytest.raises(esp.ESPError):
        rotator_mock.motor_off(axis=10)

    rotator_mock.reset_hardware()

    rotator_mock.get_position()

    rotator_mock.set_position(1)
    rotator_mock.set_position(1, relative=True)

    rotator_mock.set_velocity()
    rotator_mock.set_velocity()
    rotator_mock.set_home_velocity()
    rotator_mock.set_acceleration()
    rotator_mock.set_units(1, unit=7)
    rotator_mock.set_velocity()
    rotator_mock.set_display_resolution(axis=1, resolution=3)

    rotator_mock.close()
