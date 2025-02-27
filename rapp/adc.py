import time
import logging

import serial
import numpy as np
import struct

from rich.progress import track

from rapp.mocks import SerialMock

logger = logging.getLogger(__name__)

MAXV = 4.096
BITS = 16

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
    """Encapsulates the communication with the two-channel acquisition device (AD).

    Args:
        connection: a serial connection to the AD.
        gain: the gain to use. One of [GAIN_TWOTHIRDS, GAIN_ONE, ...].
        ch0: if true, measures the channel 0.
        ch1: if true, measures the channel 1.
        in_bytes: if true, assumes incoming data is in bytes.
        progressbar: if true, enables progress bar.

    Raises:
        ADCError: when ch0 and ch1 parameters are both False.
    """

    CMD_TEMPLATE = "{measurement};{ch0};{ch1};{samples};\n"
    AVAILABLE_CHANNELS = ['CH0', 'CH1']

    PORT = '/dev/ttyACM0'
    BAUDRATE = 57600
    TIMEOUT = 2
    TIMEOUT_OPEN = 5

    SAMPLE_RATE = 840

    def __init__(self, serial, gain=GAIN_ONE, ch0=True, ch1=True, in_bytes=True, progressbar=True,
                 timeout_open=TIMEOUT_OPEN):
        self._serial = serial
        self._in_bytes = in_bytes
        self._ch0 = ch0
        self._ch1 = ch1
        self.progressbar = ch0 != ch1
        self.timeout_open = timeout_open
        self.temperature_requested = False
        self.max_V, self._multiplier_mV = GAINS[gain]  # 5, 10/2**24 to try 24 bit ADC

        if not (ch0 or ch1):
            raise ADCError(MESSAGE_CHANNELS)

        # Arduino resets when a new serial connection is made.
        # We need to wait, otherwise we don't receive anything.
        self.wait_for_connection()

    @classmethod
    def build(cls, port=PORT, baudrate=BAUDRATE, timeout=TIMEOUT, mock_serial=False, **kwargs):
        """Builds an ADC object.

        Args:
            port: serial port in which the AD is connected.
            baudrate: bits per second.
            timeout: read timeout (seconds).
            mock_serial: if True, uses a mocked serial connection.
            kwargs: optional arguments for ADC constructor

        Returns:
            ADC: an instantiated AD object.
        """
        if mock_serial:
            logger.warning("Using mocked serial connection.")
            serial_connection = SerialMock()
        else:
            serial_connection = cls.get_serial_connection(port,
                                                          baudrate=baudrate,
                                                          timeout=timeout)

        return cls(serial_connection, **kwargs)

    @staticmethod
    def get_serial_connection(*args, **kwargs):
        try:
            return serial.Serial(*args, **kwargs)
        except serial.serialutil.SerialException as e:
            raise ADCError("Error while making connection to serial port: {}".format(e))

    def wait_for_connection(self):
        start = time.time()
        elapsed_time = 0
        queries = 0
        while elapsed_time < self.timeout_open:
            self._serial.write(b'ready?\n')
            output = self._serial.readline()
            end = time.time()
            elapsed_time = end - start
            if output == b'yes\r\n':
                break
            elif output == b'no\r\n':
                pass
            else:
                queries += 1

        if elapsed_time >= self.timeout_open:
            raise ADCError(
                "Timeout while waiting for connection to open. Timeout = {} seconds with"
                " {} failed queries.".format(self.timeout_open, queries)
            )
        logger.debug(
            "Connection opened in {} seconds with {} queries.".format(elapsed_time, queries)
        )

    def acquire(self, n_samples, flush=True):
        """Acquires voltage measurements.

        Args:
            n_samples: number of samples to acquire.
            flush: if true, flushes input from the serial port before taking measurements.

        Returns:
            the values as a list of tuples [(a00, a10), (a01, a11), ..., (a0n, a1n)].

        Raises:
            ADCError: when n_samples is not a positive number.
        """

        if n_samples <= 0:
            raise ADCError(MESSAGE_SAMPLES.format(n_samples))

        if flush:  # Clear input buffer. Otherwise messes up values at the beginning.
            self._serial.flushInput()

        cmd = ADC.CMD_TEMPLATE.format(
            measurement='adc?', ch0=int(self._ch0), ch1=int(self._ch1), samples=n_samples
        )
        logger.debug("ADC command: {}".format(cmd))

        self._serial.write(bytes(cmd, 'utf-8'))

        none_array = np.full(n_samples, None)

        ch0 = self._read_data(n_samples, name='CH0') if self._ch0 else none_array
        ch1 = self._read_data(n_samples, name='CH1') if self._ch1 else none_array

        data = list(zip(ch0, ch1))

        for ch0, ch1 in data:
            logger.debug("({}, {}) = ({}, {})".format('CH0', 'CH1', ch0, ch1))

        return data

    def request_temperature(self, flush=True):
        """Requests temperature measurements

                Args:
                    flush: if true, flushes input from the serial port before taking measurements.
                """
        if flush:  # Clear input buffer. Otherwise, messes up values at the beginning.
            self._serial.flushInput()

        cmd = "req-temp?\n"
        logger.debug("ADC command: {}".format(cmd))

        self._serial.write(bytes(cmd, 'utf-8'))
        temperature_requested = True
        return temperature_requested

    def read_temperature(self):
        """Reads temperature measurements

                Args:
                    temperature_requested:

                Returns:
                    the value as a list  [temp].
                """
        cmd = "temp?\n"
        logger.debug("ADC command: {}".format(cmd))

        self._serial.write(bytes(cmd, 'utf-8'))

        temp = self._read_data(1, name='Temperature')
        logger.debug("Temperature without correction = {}".format(temp[0]))
        temperature_requested = False
        return temp, temperature_requested

    def active_channels(self):
        """Returns number of active channels."""
        return int(self._ch0) + int(self._ch1)

    def measurement_time(self, samples):
        """Returns the time (in seconds) a measurement will take for given number of samples."""
        return samples * self.active_channels() / self.SAMPLE_RATE

    def close(self):
        self._serial.close()

    def _read_data(self, n_samples, name=''):
        data = []
        desc = "Measuring {}:".format(name)

        if name == 'CH0' or name == 'CH1':
            for _ in track(range(n_samples), description=desc, disable=not self.progressbar):
                try:
                    data.append(self._bits_to_volts(self._read_bits()))
                except (ValueError, UnicodeDecodeError) as e:
                    logger.warning("Error while reading from ADC: {}".format(e))

        if name == 'Temperature':
            data.append(self._read_float())

        return data

    def _read_bits(self):
        if self._in_bytes:
            return int.from_bytes(self._serial.read(2), byteorder='big', signed=True)
            # .read(4) for 24 bit ADC
        else:
            return int(self._serial.readline().decode().strip())

    def _bits_to_volts(self, value):
        return value * self._multiplier_mV / 1000

    def _read_float(self):
        read_value = self._serial.read(4)
        logger.debug("Temperature in bytes = {}".format(read_value))
        return struct.unpack('<f', read_value)[0]
