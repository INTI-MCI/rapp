import time
import logging

import serial
import numpy as np

from rapp.utils import progressbar

logger = logging.getLogger(__name__)


GAIN_TWOTHIRDS = 23
GAIN_ONE = 1
GAIN_TWO = 2
GAIN_FOUR = 4
GAIN_EIGHT = 8
GAIN_SIXTEEN = 16

GAINS = {
    GAIN_TWOTHIRDS: (6.144, 0.1875),
    GAIN_ONE: (4.096, 0.125),
    GAIN_TWO: (2.048, 0.0625),
    GAIN_FOUR: (1.024, 0.03125),
    GAIN_EIGHT: (0.512, 0.015625),
    GAIN_SIXTEEN: (0.256, 0.0078125),
}

MESSAGE_CHANNELS = "Both ch0 and ch1 are False. Please set at least one of them as True."
MESSAGE_SAMPLES = "You must ask for a positive number of samples... Got {}"


class ADCError(Exception):
    pass


class ADC:
    """Encapsulates the communication with the polarimeter acquisition device (AD).

    Args:
        connection: a serial connection to the AD.
        gain: the gain to use. One of [GAIN_TWOTHIRDS, GAIN_ONE, ...].
        pbar: if true, enables progressbar.
    """

    COMMAND_TEMPLATE = "{ch0};{ch1};{samples}s"
    AVAILABLE_CHANNELS = ['CH0', 'CH1']

    PORT = '/dev/ttyACM0'
    BAUDRATE = 57600
    TIMEOUT = 0.1
    WAIT = 2

    def __init__(self, connection, gain=GAIN_ONE, progressbar=True):
        self._connection = connection
        self.progressbar = progressbar
        self.max_V, self._multiplier_mV = GAINS[gain]

    @classmethod
    def build(cls, port=PORT, b=BAUDRATE, timeout=TIMEOUT, wait=WAIT, **kwargs):
        """Builds an ADC object.

        Args:
            port: serial port in which the AD is connected.
            baudrate: bits per second.
            timeout: read timeout (seconds).
            wait: time to wait after making a connection (seconds).
            kwargs: optinal arguments for ADC constructor

        Returns:
            ADC: an instantiated AD object.
        """
        connection = cls.serial_connection(port, baudrate=b, timeout=timeout)

        logger.info("Waiting {} seconds after connecting to ADC...".format(wait))
        # Arduino resets when a new serial connection is made.
        # We need to wait, otherwise we don't recieve anything.
        # TODO: check if we can avoid that arduino resets.
        time.sleep(wait)

        return cls(connection, **kwargs)

    @staticmethod
    def serial_connection(*args, **kwargs):
        try:
            return serial.Serial(*args, **kwargs)
        except serial.serialutil.SerialException as e:
            raise ADCError("Error while making connection to serial port: {}".format(e))

    def acquire(self, n_samples, ch0=True, ch1=True, in_bytes=True):
        """Acquires data from the AD.

        Args:
            n_samples: number of samples to ask.
            in_bytes: if true, AD will send values as bytes. If false, as text.
            ch0: if true, measures the channel 0.
            ch1: if true, measures the channel 1.

        Returns:
            the values as a list of tuples [(a00, a10), (a01, a11), ..., (a0n, a1n)].

        Raises:
            ADCError: when ch0 and ch1 parameters are both False.
        """

        if not (ch0 or ch1):
            raise ADCError(MESSAGE_CHANNELS)

        if n_samples <= 0:
            raise ADCError(MESSAGE_SAMPLES.format(n_samples))

        adc_command = ADC.COMMAND_TEMPLATE.format(ch0=int(ch0), ch1=int(ch1), samples=n_samples)
        logger.debug("ADC command: {}".format(adc_command))

        self._connection.write(bytes(adc_command, 'utf-8'))

        none = np.full(n_samples, None)

        data = {}
        data['CH0'] = self._read_data(n_samples, in_bytes=in_bytes, name='CH0') if ch0 else none
        data['CH1'] = self._read_data(n_samples, in_bytes=in_bytes, name='CH1') if ch1 else none

        return data

    def flush_input(self):
        self._connection.flushInput()

    def close(self):
        self._connection.close()

    def _read_data(self, n_samples, in_bytes=True, name=''):
        data = []
        for _ in progressbar(range(n_samples), prefix="{}:".format(name), enable=self.progressbar):
            try:
                if in_bytes:
                    value = self._connection.read(2)
                    value = int.from_bytes(value, byteorder='big', signed=True)
                else:
                    value = self._connection.readline().decode().strip()
                    value = int(value)

                data.append(self._bits_to_volts(value))
            except (ValueError, UnicodeDecodeError) as e:
                logger.warning("Error while reading from ADC: {}".format(e))

        return data

    def _bits_to_volts(self, value):
        return value * self._multiplier_mV / 1000
