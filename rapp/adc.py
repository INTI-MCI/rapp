import time
import logging

import serial
import numpy as np

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


class ADCError(Exception):
    pass


class ADC:
    """Encapsulates the communication with the polarimeter acquisition device (AD).

    Args:
        dev: the device of the serial in which the AD is connected.
        baudrate: bits per second.
        timeout: read timeout (seconds).
        gain: the gain to use. One of [GAIN_TWOTHIRDS, GAIN_ONE, ...].
        wait: time to wait after making a connection (seconds).
    """

    COMMAND_TEMPLATE = "{ch0};{ch1};{samples}s"
    CHANNEL_NAME = "C{}"

    def __init__(self, dev, b=57600, timeout=0.1, gain=GAIN_ONE, wait=2):
        self._serial = serial.Serial(dev, baudrate=b, timeout=timeout)

        logger.info("Waiting {} seconds after opening the connection...".format(wait))
        # Arduino resets when a new serial connection is made.
        # We need to wait, otherwise we don't recieve anything.
        # TODO: check if we can avoid that arduino resets.
        time.sleep(wait)

        self._max_V, self._multiplier_mV = GAINS[gain]

    def acquire(self, n_samples, ch0=True, ch1=True, in_bytes=True):
        """Acquires data from the AD.

        Args:
            n_samples: number of samples to ask.
            in_bytes: if true, AD will send values as bytes. If false, as text.
            ch0: if true, measures the channel 0.
            ch1: if true, measures the channel 1.

        Returns:
            the values as a list of tuples [(a00, a10), (a01, a11), ..., (a0n, a1n)].
        """

        if not (ch0 or ch1):
            raise ADCError(MESSAGE_CHANNELS)

        adc_command = ADC.COMMAND_TEMPLATE.format(ch0=int(ch0), ch1=int(ch1), samples=n_samples)
        logger.debug("ADC command: {}".format(adc_command))

        self._serial.write(bytes(adc_command, 'utf-8'))

        data = {}
        data['C0'] = self._read_data(n_samples, in_bytes=True) if ch0 else np.full(n_samples, None)
        data['C1'] = self._read_data(n_samples, in_bytes=True) if ch1 else np.full(n_samples, None)

        return data

    def flush_input(self):
        self._serial.flushInput()

    def close(self):
        self._serial.close()

    def _read_data(self, n_samples, in_bytes=True):
        data = []
        for _ in range(n_samples):
            try:
                if in_bytes:
                    value = self._serial.read(2)
                    value = int.from_bytes(value, byteorder='big', signed=True)
                else:
                    value = self._serial.readline().decode().strip()
                    value = int(value)

                data.append(self._bits_to_volts(value))
            except (ValueError, UnicodeDecodeError) as e:
                logger.warning("Error while reading from ADC: {}".format(e))

        return data

    def _bits_to_volts(self, value):
        return value * self._multiplier_mV / 1000
