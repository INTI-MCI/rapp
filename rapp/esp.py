import logging
import serial

logger = logging.getLogger(__name__)


ERROR_SETUP = "Error while setting up controller, error #{}"
ERROR_ALLOWED_AXES = "Specified axis {} is not one of the allowed axes: {}"


class ESPError(Exception):
    pass


class ESP:
    """Encapsulates the communication with the rotary motion controller.

    Args:
        connection: a serial connection to the motion controller.
        axis: axis to use in every command unless specified otherwise.
        initpos: initial position to use.
        useaxis: list of axis that are going to be used.
    """

    ALLOWED_AXES = (1, 2, 3)
    PORT = '/dev/ttyACM0'
    BAUDRATE = 19200

    def __init__(self, connection, axis=1, reset=False, initpos=None, useaxes=None):
        self._connection = connection

        self.axes_in_use = useaxes

        if self.axes_in_use is None:
            self.axes_in_use = [axis]

        self.default_axis = axis

        logger.debug("Turning motor ON for axis {}...".format(axis))
        self.motor_on(axis=axis)

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
        """Builds an ADC object.

        Args:
            port: serial port in which the ESP controller is connected.
            baudrate: bits per second.
            kwargs: optinal arguments for ESP constructor

        Returns:
            ESP: an instantiated ESP object.
        """
        connection = cls.serial_connection(port, baudrate=b)
        return cls(connection, **kwargs)

    @staticmethod
    def serial_connection(*args, **kwargs):
        try:
            return serial.Serial(*args, **kwargs)
        except serial.serialutil.SerialException as e:
            raise ESPError("Error while making connection to serial port: {}".format(e))

    def motor_on(self, axis=None):
        axis = self._resolve_axis(axis)
        self._connection.write("{0}MO; {0}MO?\r".format(axis).encode())
        res = float(self._connection.readline())

        return res == 1

    def motor_off(self, axis=None):
        axis = self._resolve_axis(axis)
        self._connection.write("{0}MF; {0}MF?\r".format(axis).encode())
        res = float(self._connection.readline())

        return res == 1

    def reset_hardware(self):
        self._connection.write("RS\r".encode())

    def reset_axis(self, axis):
        self._connection.write("{0}OR;{0}WS0\r".format(axis).encode())

    def get_position(self, axis=None):
        axis = self._resolve_axis(axis)
        self._connection.write("{0}TP\r".format(axis).encode())

        return float(self._connection.readline())

    def set_position(self, position, ws=0, axis=None, relative=False):
        axis = self._resolve_axis(axis)

        position_type = "PA"
        if relative:
            position_type = "PR"

        command_tpl = "{0}{type}{1:.4f};{0}WS{ws};{0}WP{1:.4f};{0}TP\r"
        cmd = command_tpl.format(axis, position, type=position_type, ws=ws)

        self._connection.write(cmd.encode())
        return float(self._connection.readline())

    def set_velocity(self, vel=2, axis=None):
        axis = self._resolve_axis(axis)
        self._connection.write("{0}VA{1:.4f};{0}TV\r".format(axis, vel).encode())

        return float(self._connection.readline())

    def set_home_velocity(self, vel=2, axis=None):
        axis = self._resolve_axis(axis)
        self._connection.write("{0}OL{1}\r".format(axis, vel).encode())
        self._connection.write("{0}OH{1}; {0}OH?\r".format(axis, vel).encode())

        return float(self._connection.readline())

    def set_acceleration(self, acc=2, axis=None):
        axis = self._resolve_axis(axis)
        self._connection.write("{0}CA{1:.4f};\r".format(axis, acc).encode())

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
        self._connection.write("{0}SN{1}\r".format(axis, unit).encode())

    def set_display_resolution(self, axis, resolution):
        """Sets amount of decimals in user units."""
        self._connection.write("{0}FP{1}\r".format(axis, resolution).encode())

    def close(self):
        self._connection.close()

    def _check_errors(self):
        self._connection.write("TE?\r".encode())
        output = int(self._connection.readline())

        if (output != 0):
            raise ESPError(ERROR_SETUP.format(output))

    def _resolve_axis(self, axis=None):
        a = self.default_axis

        if axis is not None:
            if axis not in ESP.ALLOWED_AXES:
                raise ESPError(ERROR_ALLOWED_AXES.format(axis, ESP.ALLOWED_AXES))
            a = axis

        return a
