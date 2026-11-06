import logging
import struct
import time

import numpy as np
import serial as serial_lib
from rich.progress import track

from rapp.mocks import SerialMock

logger = logging.getLogger(__name__)

MAXV = 4.096
BITS = 16  # TODO: Check usages

GAIN_TWOTHIRDS = 23
GAIN_ONE = 1
GAIN_TWO = 2
GAIN_FOUR = 4
GAIN_EIGHT = 8
GAIN_SIXTEEN = 16

SUPPORTED_ADCS = {
    "CS_1237": {
        'BITS': 24,
        'N_BYTES': 4,
        'BYTE_ORDER': 'big',
        'GAINS': {
            1: (1.24, 1240/2**23),
            2: (0.62, 620/2**23),
            64: (0.019375, 19.375/2**23),
            128: (0.0096875, 9.6875/2**23)
        }

    },
    "ADS_1115": {
        'BITS': 16,
        'N_BYTES': 2,
        'BYTE_ORDER': 'big',  # little or big
        'GAINS': {
            GAIN_TWOTHIRDS: (6.144, 0.1875),
            GAIN_ONE: (4.096, 0.125),
            GAIN_TWO: (2.048, 0.0625),
            GAIN_FOUR: (1.024, 0.03125),
            GAIN_EIGHT: (0.512, 0.015625),
            GAIN_SIXTEEN: (0.256, 0.0078125)}
    },
    "ADS_1232": {
        'BITS': 24,
        'N_BYTES': 4,
        'BYTE_ORDER': 'big',
        'GAINS': {
            1: (2.49, 2490/2**23),
            2: (1.25, 1250/2**23),
            64: (0.039, 39/2**23),
            128: (0.0235, 23.5/2**23),
        }
    }
}

MESSAGE_CHANNELS = "Both ch0 and ch1 are False. Please set at least one of them as True."
MESSAGE_SAMPLES = "You must ask for a positive number of samples... Got {}"
MIN_32BIT_INT = -2147483648
MAX_ERROR_CODE = MIN_32BIT_INT + 10
ERROR_CODES = {
    0: 'NoERROR',
    1: 'Timeout waiting for HIGH',
    2: 'Timeout waiting for LOW',
	5: 'DIVIDED_by_ZERO',
}

class ADCError(Exception):
    pass

class ADCBadCommandError(ADCError):
    pass

class ADCRecoveryError(ADCError):
    pass


class ADC:
    """Encapsulates the communication with the two-channel acquisition device (AD).

    Args:
        connection: a serial connection to the AD.
        gain: the gain to use. One of GAINS for 16 bit adc or PGA for 24 bit adc.
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
    ADC_WRITE_TIMEOUT = 2
    TIMEOUT_OPEN = 7

    SAMPLE_RATE = 840
    SAMPLE_RATE_24 = 1280

    def __init__(self, serial, adc_used, gain=1, ch0=True, ch1=True, in_bytes=True, progressbar=True,
                 timeout_open=TIMEOUT_OPEN):
        self._serial: serial_lib.Serial = serial
        self._in_bytes = in_bytes
        self._ch0 = ch0
        self._ch1 = ch1
        self.progressbar = ch0 != ch1
        self.timeout_open = timeout_open
        self.temperature_requested = False
        self.bits = SUPPORTED_ADCS[adc_used]["BITS"]
        self.max_V, self._multiplier_mV = SUPPORTED_ADCS[adc_used]["GAINS"][gain]
        self._n_bytes = SUPPORTED_ADCS[adc_used]["N_BYTES"]
        self._byte_order = SUPPORTED_ADCS[adc_used]["BYTE_ORDER"]

        if not (ch0 or ch1):
            raise ADCError(MESSAGE_CHANNELS)
        # Arduino resets when a new serial connection is made.
        # We need to wait, otherwise we don't receive anything.
        self.wait_for_connection()

    @classmethod
    def build(cls, port=PORT, baudrate=BAUDRATE, timeout=TIMEOUT, write_timeout=ADC_WRITE_TIMEOUT,
              mock_serial=False, **kwargs):
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
                                                          timeout=timeout,
                                                          write_timeout=write_timeout)

        return cls(serial_connection, **kwargs)

    @staticmethod
    def get_serial_connection(*args, **kwargs):
        try:
            return serial_lib.Serial(*args, **kwargs)
        except serial_lib.SerialException as e:
            raise ADCError("Error while making connection to serial port: {}".format(e))

    def _send_command(self, cmd: str, error_message="Timeout while writing to serial port"):
        self._serial_write(cmd, error_message)
        line = self._serial.readline()  # Read ACK from Arduino
        if line != b'ACK\r\n':
            raise ADCBadCommandError("Expected 'ACK' but received '{}'".format(line))

    def _serial_write(self, cmd, error_message="Timeout while writing to serial port"):
        try:
            if isinstance(cmd, str):
                cmd = bytes(cmd, 'utf-8')
            self._serial.write(cmd)
        except serial_lib.SerialTimeoutException as e:
            raise ADCError("{}: {}".format(error_message, e))

    def wait_for_connection(self):
        start = time.time()
        elapsed_time = 0
        queries = 0
        while elapsed_time < self.timeout_open:
            self._serial_write('ready?\n')
            output = self._serial.readline()
            end = time.time()
            elapsed_time = end - start
            if output == b'yes\r\n':
                line = self._serial.readline()
                if line:
                    logger.info("Data in input buffer after making connection: {}".format(line))
                self._serial.reset_input_buffer()  # TODO: check if this is necessary
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

    def acquire(self, n_samples, flush=True, hex=False):
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
            self._serial.reset_input_buffer()

        cmd = ADC.CMD_TEMPLATE.format(
            measurement='adc', ch0=int(self._ch0), ch1=int(self._ch1), samples=n_samples
        )
        logger.debug("ADC command: {}".format(cmd))
        self._send_command(cmd)

        none_array = np.full(n_samples, None)

        if self._ch0:
            ch0 = self._read_data(n_samples, name='CH0', hex=hex)
            if len(ch0) != n_samples:
                return None
        else:
            ch0 = none_array
        if self._ch1:
            ch1 = self._read_data(n_samples, name='CH1', hex=hex)
            if len(ch1) != n_samples:
                return None
        else:
            ch1 = none_array

        data = list(zip(ch0, ch1))

        for ch0, ch1 in data:
            logger.debug("({}, {}) = ({}, {})".format('CH0', 'CH1', ch0, ch1))

        return data

    def request_temperature(self, channel, flush=True):
        """Requests temperature measurements

                Args:
                    flush: if true, flushes input from the serial port before taking measurements.
                    channel: channel 0 - Room temperature, channel 1 - Quartz plate temperature
                """
        if flush:  # Clear input buffer. Otherwise, messes up values at the beginning.
            self._serial.reset_input_buffer()

        cmd = f"req-temp;{channel};\n"
        logger.debug("ADC command: {}".format(cmd))

        self._send_command(cmd)

        temperature_requested = True
        return temperature_requested

    def read_temperature(self, channel=0):  # TODO: ver si saco el valor por default de channel
        """Reads temperature measurements

                Args:
                    channel: channel 0 - Room temperature, channel 1 - Quartz plate temperature

                Returns:
                    the value as a list  [temp].
                """
        cmd = f"temp;{channel};\n"
        logger.debug("ADC command: {}".format(cmd))

        self._send_command(cmd)

        temp = self._read_data(1, name='Temperature')
        logger.debug("Temperature without correction = {}".format(temp[0]))
        temperature_requested = False
        return temp, temperature_requested

    def active_channels(self):
        """Returns number of active channels."""
        return int(self._ch0) + int(self._ch1)

    def measurement_time(self, samples):
        """Returns the time (in seconds) a measurement will take for given number of samples."""
        return samples * self.active_channels() / self.SAMPLE_RATE_24

    def close(self):
        self._serial.close()

    def recover(self):
        """Resets the Arduino connection and calibrates the ADC after a read failure."""
        try:
            self._serial.dtr = False
            time.sleep(0.1)
            self._serial.dtr = True
            self.wait_for_connection()
            self.self_calibrate()
        except (ADCError, serial_lib.SerialException, AttributeError) as e:
            raise ADCRecoveryError("Could not recover ADC communication: {}".format(e)) from e

    def _read_data(self, n_samples, name='', hex=False):
        data = []
        desc = "Measuring {}:".format(name)

        if name == 'CH0' or name == 'CH1':
            for _ in track(range(n_samples), description=desc, disable=not self.progressbar):
                try:
                    measurement = self._read_bits()
                    if MIN_32BIT_INT < measurement <= MAX_ERROR_CODE:
                        error_code = measurement - MIN_32BIT_INT
                        raise ADCError("ADC error: {}".format(ERROR_CODES[error_code]))
                    if not hex:
                        measurement = self._bits_to_volts(measurement)
                    if measurement is None:
                        break
                    data.append(measurement)
                except (ValueError, UnicodeDecodeError) as e:
                    logger.warning("Error while reading from ADC: {}".format(e))

        if name == 'Temperature':
            data.append(self._read_float())

        return data

    def _read_bits(self):
        if self._in_bytes:
            bytes_data = self._read_bytes(self._n_bytes)
            str_bytes = bytes_data.hex()
            logger.debug('Data in bytes = %s', bytes_data)
            return int.from_bytes(bytes_data, byteorder=self._byte_order, signed=True)
        else:
            return int(self._serial.readline().decode().strip())

    def _read_bytes(self, n_bytes):
        bytes_data = b""

        for _ in range(3):
            try:
                bytes_data += self._serial.read(n_bytes - len(bytes_data))
            except serial_lib.SerialException as e:
                raise ADCError(f"Error while reading from serial port: {e}") from e
            if len(bytes_data) == n_bytes:
                return bytes_data

        logger.warning(
            "ADCError: Expected {} bytes, got {} bytes.".format(n_bytes, len(bytes_data))
        )
        raise ADCError("Expected {} bytes, got {} bytes.".format(n_bytes, len(bytes_data)))

    def _bits_to_volts(self, value):
        if value is None:
            return None
        return value * self._multiplier_mV / 1000

    def _read_float(self):
        read_value = self._read_bytes(4)
        logger.debug("Temperature in bytes = {}".format(read_value))
        return struct.unpack('<f', read_value)[0]

    def print_read_n_every_dt(self, channel, n_samples, dt_mus, flush=True):
        """Prints the values acquired by the ADC.

        Args:
            channel: channel to acquire.
            n_samples: number of samples to acquire.
            dt_mus: time between samples in microseconds.
            flush: if true, flushes input from the serial port before taking measurements.
        """
        if n_samples <= 0:
            raise ADCError(MESSAGE_SAMPLES.format(n_samples))

        if flush:  # Clear input buffer. Otherwise messes up values at the beginning.
            self._serial.reset_input_buffer()

        cmd = "get_n_every_dt;{channel};{samples};{dt_mus};\n".format(channel=channel, samples=n_samples, dt_mus=dt_mus)
        logger.debug("ADC command: {}".format(cmd))

        self._send_command(cmd)
        times = np.zeros(n_samples)

        for n in range(n_samples):
            times[n] = time.time()
            measurement = self._bits_to_volts(self._read_bits())
            if measurement is None:
                break
            logger.info(f"{n}: {measurement}")
        mean_dt_us = np.mean(np.diff(times)) * 1e6
        return mean_dt_us

    def self_calibrate(self):
        """Sends a command to the ADC to perform self-calibration."""
        cmd = "self_calibrate;\n"
        logger.debug("ADC command: {}".format(cmd))

        self._send_command(cmd)

        # Read two uint16 values
        err0 = struct.unpack('<H', self._read_bytes(2))[0]
        err1 = struct.unpack('<H', self._read_bytes(2))[0]
        if err0 != 0:
            logger.error(f"Error during self-calibration of ADC0: {err0}")
        if err1 != 0:
            logger.error(f"Error during self-calibration of ADC1: {err1}")

    def reset_ADC(self):
        """Sends a command to the ADC to perform a reset."""
        cmd = "reset_adc;\n"
        logger.debug("ADC command: {}".format(cmd))

        self._send_command(cmd)
        self._serial.reset_input_buffer()