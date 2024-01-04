import logging
import serial

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
        initpos: initial position to use.
        useaxis: list of axis that are going to be used.
    """

    ALLOWED_AXES = (1, 2, 3)
    PORT = '/dev/ttyACM0'
    BAUDRATE = 19200

    def __init__(self, serial, axis=1, reset=False, initpos=None, useaxes=None):
        self._serial = serial

        self.axes_in_use = useaxes

        if self.axes_in_use is None:
            self.axes_in_use = [axis]

        self.default_axis = axis

        if reset:
            logger.info("Resetting axes in use: {}".format(self.axes_in_use))
            for n in self.axes_in_use:
                self.reset_axis(n)
                self._check_errors()

        if initpos is not None and initpos != 0:
            logger.info("Setting initial position: {}".format(initpos))
            for n in self.axes_in_use:
                self.set_position(initpos)
                self._check_errors()

    @classmethod
    def build(cls, port=PORT, b=BAUDRATE, **kwargs):
        """Builds an ESP object.

        Args:
            port: serial port in which the ESP controller is connected.
            baudrate: bits per second.
            kwargs: optinal arguments for ESP constructor

        Returns:
            ESP: an instantiated ESP object.
        """
        serial_connection = cls.get_serial_connection(port, baudrate=b)
        return cls(serial_connection, **kwargs)

    @staticmethod
    def get_serial_connection(*args, **kwargs):
        try:
            return serial.Serial(*args, **kwargs)
        except serial.serialutil.SerialException as e:
            raise ESP301Error("Error while making connection to serial port: {}".format(e))

    def motor_on(self, axis=None):
        axis = self._resolve_axis(axis)
        self._serial.write("{0}MO; {0}MO?\r".format(axis).encode())
        res = float(self._serial.readline())

        return res == 1

    def motor_off(self, axis=None):
        axis = self._resolve_axis(axis)
        self._serial.write("{0}MF; {0}MF?\r".format(axis).encode())
        res = float(self._serial.readline())

        return res == 1

    def reset_hardware(self):
        self._serial.write("RS\r".encode())

    def reset_axis(self, axis):
        self._serial.write("{0}OR;{0}WS0\r".format(axis).encode())

    def get_position(self, axis=None):
        axis = self._resolve_axis(axis)
        self._serial.write("{0}TP\r".format(axis).encode())

        return float(self._serial.readline())

    def set_position(self, position, ws=0, axis=None, relative=False):
        axis = self._resolve_axis(axis)

        position_type = "PA"
        if relative:
            position_type = "PR"

        command_tpl = "{0}{type}{1:.4f};{0}WS{ws};{0}WP{1:.4f};{0}TP\r"
        cmd = command_tpl.format(axis, position, type=position_type, ws=ws)

        self._serial.write(cmd.encode())
        return float(self._serial.readline())

    def set_velocity(self, velocity, axis=None):
        axis = self._resolve_axis(axis)
        self._serial.write("{0}VA{1:.4f};{0}TV\r".format(axis, velocity).encode())

        return float(self._serial.readline())

    def set_home_velocity(self, velocity, axis=None):
        axis = self._resolve_axis(axis)
        self._serial.write("{0}OL{1}\r".format(axis, velocity).encode())
        self._serial.write("{0}OH{1}; {0}OH?\r".format(axis, velocity).encode())

        return float(self._serial.readline())

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

    def close(self):
        self._serial.close()

    def _check_errors(self):
        self._serial.write("TE?\r".encode())
        output = int(self._serial.readline())

        if (output != 0):
            raise ESP301Error(ERROR_SETUP.format(output))

    def _resolve_axis(self, axis=None):
        _axis = self.default_axis

        if axis is not None:
            if axis not in ESP301.ALLOWED_AXES:
                raise ESP301Error(ERROR_ALLOWED_AXES.format(axis, ESP301.ALLOWED_AXES))

            _axis = axis

        return _axis
