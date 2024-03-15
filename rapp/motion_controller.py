import logging

import serial

from rapp.mocks import SerialMock

logger = logging.getLogger(__name__)


ERROR_SETUP = "Error while setting up controller, error #{}"
ERROR_ALLOWED_AXES = "Specified axis {} is not one of the allowed axes: {}"


class ESP301Error(Exception):
    pass


class ESP301:
    """Encapsulates the communication with the motion controller Newport ESP301.

    Args:
        serial: serial connection to the motion controller.
        axis: axis to use in every command unless specified otherwise.
        useaxis: list of axis that are going to be used.
    """

    ALLOWED_AXES = (1, 2, 3)
    PORT = '/dev/ttyACM0'
    BAUDRATE = 19200

    def __init__(self, serial, axis=1, useaxes=None):
        self._serial = serial

        self.axes_in_use = useaxes

        if self.axes_in_use is None:
            self.axes_in_use = [axis]

        self.default_axis = axis

        for n in self.axes_in_use:
            logger.info("Setting motor ON for axis: {}".format(n))
            self.motor_on(axis=n)

    @classmethod
    def build(cls, port=PORT, b=BAUDRATE, mock_serial=False, **kwargs):
        """Builds an ESP301 object.

        Args:
            port: serial port in which the ESP301 controller is connected.
            baudrate: bits per second.
            mock_serial: if True, uses a mocked serial connection.
            kwargs: optional arguments for the constructor.

        Returns:
            ESP301: an instantiated ESP301 object.
        """
        if mock_serial:
            logger.warning("Using mocked serial connection.")
            return cls(SerialMock(), **kwargs)

        serial_connection = cls.get_serial_connection(port, baudrate=b)
        return cls(serial_connection, **kwargs)

    @staticmethod
    def get_serial_connection(*args, **kwargs):
        try:
            return serial.Serial(*args, **kwargs)
        except serial.serialutil.SerialException as e:
            raise ESP301Error("Error while making connection to serial port: {}".format(e))

    def close(self):
        """Closes the connection."""
        self._serial.close()

    def check_errors(self):
        """Reads internal buffer of the device. If an error is found, raises an exception."""
        self._serial.write("TE?\r".encode())
        output = int(self._read_serial())

        if (output != 0):
            raise ESP301Error(ERROR_SETUP.format(output))

    def get_error(self):
        """Reads error from the device buffer."""
        self._serial.write("TB?\r".encode())

        return self._serial.readline()

    def get_position(self, axis=None):
        axis = self._resolve_axis(axis)
        self._serial.write("{0}TP\r".format(axis).encode())

        return self._read_float()

    def motor_on(self, axis=None):
        axis = self._resolve_axis(axis)
        self._serial.write("{0}MO; {0}MO?\r".format(axis).encode())
        res = self._read_float()

        return res == 1

    def motor_off(self, axis=None):
        axis = self._resolve_axis(axis)
        self._serial.write("{0}MF; {0}MF?\r".format(axis).encode())
        res = self._read_float()

        return res == 1

    def set_home(self, position, axis=None):
        axis = self._resolve_axis(axis)

        command_tpl = "{0}DH{1:.4f}\r"
        cmd = command_tpl.format(axis, position)

        self._serial.write(cmd.encode())

    def set_position(self, position, ws=0, axis=None, relative=False):
        axis = self._resolve_axis(axis)

        position_type = "PA"
        if relative:
            position_type = "PR"

        command_tpl = "{0}{type}{1:.4f};{0}WS{ws};{0}WP{1:.4f};{0}TP\r"
        cmd = command_tpl.format(axis, position, type=position_type, ws=ws)

        self._serial.write(cmd.encode())

        return self._read_float()

    def set_velocity(self, velocity, axis=None):
        axis = self._resolve_axis(axis)
        self._serial.write("{0}VA{1:.4f};{0}TV\r".format(axis, velocity).encode())

        return self._read_float()

    def set_home_velocity(self, velocity, axis=None):
        axis = self._resolve_axis(axis)
        self._serial.write("{0}OL{1}\r".format(axis, velocity).encode())
        self._serial.write("{0}OH{1}; {0}OH?\r".format(axis, velocity).encode())

        return self._read_float()

    def set_acceleration(self, acceleration, axis=None):
        axis = self._resolve_axis(axis)
        self._serial.write("{0}CA{1:.4f};\r".format(axis, acceleration).encode())

    def set_units(self, axis, unit=7):
        """Set axis displacement units.

        Args:
            axis: axis to change.
            unit: 0 to 10 where
                0 = encoder count
                1 = motor step
                2 = millimeter
                3 = micrometer
                4 = inches
                5 = milli-inches
                or ? to read present setting
                6 = micro-inches
                7 = degree
                8 = gradian
                9 = radian
                10 = milliradian
                11 = microradian
        """
        self._serial.write("{0}SN{1}\r".format(axis, unit).encode())

    def set_display_resolution(self, axis, resolution):
        """Sets amount of decimals in user units."""
        self._serial.write("{0}FP{1}\r".format(axis, resolution).encode())

    def reset_hardware(self):
        self._serial.write("RS\r".encode())

    def reset_axis(self, axis):
        self._serial.write("{0}OR;{0}WS0\r".format(axis).encode())

    def reconnect(self):
        """Reconnects to the serial port."""

        self._serial.close()
        self._serial.open()

    def _resolve_axis(self, axis=None):
        _axis = self.default_axis

        if axis is not None:
            if axis not in ESP301.ALLOWED_AXES:
                raise ESP301Error(ERROR_ALLOWED_AXES.format(axis, ESP301.ALLOWED_AXES))

            _axis = axis

        return _axis

    def _read_serial(self):
        response = self._serial.readline()
        if response == b'':
            raise ESP301Error("Found empty buffer trying to read serial data.")

        return response

    def _read_float(self):
        res = self._read_serial()
        try:
            return float(res)
        except ValueError:
            raise ESP301Error(
                "Bad response from motion controller: {}. "
                "Expected float, received: {}.".format(res, type(res)))
