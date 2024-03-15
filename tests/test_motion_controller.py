from rapp.motion_controller import ESP301, ESP301Error
from rapp import mocks

import pytest  # noqa
import serial


class ErrorSerialMock(mocks.SerialMock):
    def readline(*args, **kwargs):
        return ''


def test_init():
    serial_mock = mocks.SerialMock()

    ESP301(serial_mock)

    ESP301(mocks.SerialMock())
    rotator = ESP301(serial_mock, useaxes=[1, 2])

    assert rotator.axes_in_use == [1, 2]


def test_build(monkeypatch):
    with pytest.raises(ESP301Error):
        ESP301.build(dev='BADDEVICE')

    monkeypatch.setattr(serial, "Serial", lambda *x, **y: mocks.SerialMock())
    ESP301.build()


def test_methods(motion_controller):
    motion_controller.motor_off(axis=2)

    with pytest.raises(ESP301Error):
        motion_controller.motor_off(axis=10)

    motion_controller.reset_hardware()

    motion_controller.get_position()

    motion_controller.set_position(1)
    motion_controller.set_position(1, relative=True)

    motion_controller.set_velocity(4)
    motion_controller.set_home_velocity(4)
    motion_controller.set_acceleration(4)
    motion_controller.set_units(1, unit=7)
    motion_controller.set_display_resolution(axis=1, resolution=3)

    with pytest.raises(ESP301Error):
        ESP301(ErrorSerialMock())

    motion_controller.close()
