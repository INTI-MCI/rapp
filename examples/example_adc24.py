import sys
import time
import serial
import numpy as np

from rapp.utils import timing
from rapp.adc import ADC

from rich.progress import track


ADC_WIN_DEVICE = 'COM3'
ADC_LINUX_DEVICE = '/dev/ttyACM0'
ADC_BAUDRATE = 57600
ADC_TIMEOUT = 2
ADC_TIMEOUT_OPEN = 5

ADC_MULTIPLIER_mV = 0.125
ADC_WAIT_TIME = 5

CMD_TEMPLATE = "{measurement};{ch0};{ch1};{samples};\n"
AVAILABLE_CHANNELS = ['CH0', 'CH1']

_in_bytes = True

class ADCError(Exception):
    pass


def get_serial_connection(*args, **kwargs):
    try:
        return serial.Serial(*args, **kwargs)
    except serial.serialutil.SerialException as e:
        raise ADCError("Error while making connection to serial port: {}".format(e))


def wait_for_connection(self):
    start = time.time()
    elapsed_time = 0
    queries = 0
    while elapsed_time < ADC_TIMEOUT_OPEN:
        self.write(b'ready?\n')
        output = self.readline()
        end = time.time()
        elapsed_time = end - start
        print('output: ', output)
        if output == b'yes\r\n':
            line = self.readline()
            self.reset_input_buffer()
            print("Line: {}".format(line))
            # if self.inWaiting():
            #     print("Connection opened, buffer: {}".format(self.inWaiting()))
            #     self.reset_input_buffer()
            #     print("Connection opened, buffer: {}".format(self.inWaiting()))
            #     buf = self.read(self.inWaiting())
            #     line = self.readline()
            #     print("Buffer: {}, line: {}".format(buf, line))
            break
        elif output == b'no\r\n':
            pass
        else:
            queries += 1

    if elapsed_time >= ADC_TIMEOUT_OPEN:
        print(
            "Timeout while waiting for connection to open. Timeout = {} seconds with"
            " {} failed queries.".format(ADC_TIMEOUT_OPEN, queries)
        )
    print(
        "Connection opened in {} seconds with {} queries.".format(elapsed_time, queries)
    )


def acquire(self, n_samples, _ch0=True, _ch1=True, flush=True):
    """Acquires voltage measurements.

    Args:
        n_samples: number of samples to acquire.
        flush: if true, flushes input from the serial port before taking measurements.

    Returns:
        the values as a list of tuples [(a00, a10), (a01, a11), ..., (a0n, a1n)].

    Raises:
        ADCError: when n_samples is not a positive number.
    """

    if flush:  # Clear input buffer. Otherwise messes up values at the beginning.
        self.flushInput()

    cmd = CMD_TEMPLATE.format(
        measurement='adc', ch0=int(_ch0), ch1=int(_ch1), samples=n_samples
    )
    print("ADC command: {}".format(cmd))

    self.write(bytes(cmd, 'utf-8'))

    none_array = np.full(n_samples, None)

    ch0 = self._read_data(n_samples, name='CH0') if _ch0 else none_array
    ch1 = self._read_data(n_samples, name='CH1') if _ch1 else none_array

    data = list(zip(ch0, ch1))

    for ch0, ch1 in data:
        print("({}, {}) = ({}, {})".format('CH0', 'CH1', ch0, ch1))

    return data


def _read_data(self, n_samples, name=''):
    data = []
    desc = "Measuring {}:".format(name)

    if name == 'CH0' or name == 'CH1':
        for _ in track(range(n_samples), description=desc, disable=not self.progressbar):
            try:
                data.append(_read_bits(self))# (self._bits_to_volts(self._read_bits()))
            except (ValueError, UnicodeDecodeError) as e:
                print("Error while reading from ADC: {}".format(e))

    if name == 'Temperature':
        data.append(_read_float(self))

    return data


def _read_bits(self):
    if _in_bytes:
        datos = self.read(4)
        print(datos)
        return int.from_bytes(datos, byteorder='big', signed=True)
    else:
        return int(self.readline().decode().strip())


def main(n_samples=5, ch0=1, ch1=1):
    print("Instantiating ADC...")
    adc = get_serial_connection(ADC_WIN_DEVICE, baudrate=ADC_BAUDRATE, timeout=ADC_TIMEOUT)
    # adc = ADC(resolve_adc_device(), timeout_open=ADC_TIMEOUT_OPEN)#, baudrate=ADC_BAUDRATE, timeout=ADC_TIMEOUT)
    wait_for_connection(adc)
    # buf = adc.readline()
    # print(buf)
    adc.reset_input_buffer()
    # time.sleep(5)
    # buf2 = adc.readline()
    # print(buf2)
    adc.write(bytes(CMD_TEMPLATE.format(measurement='adc', ch0=ch0, ch1=ch1, samples=n_samples).encode('utf-8')))
    # adc.write(bytes('adc_n_dt;{};500;\n'.format(n_samples).encode('utf-8')))
    none_array0 = np.full(n_samples, None)
    none_array1 = np.full(n_samples, None)
    datos0 = []
    datos1 = []

    for i in range(n_samples):
        channel0 = adc.read(4) if ch0 else none_array0
        datos0.append(channel0)
        datos0.append(int.from_bytes(channel0, byteorder='big', signed=True))

    for i in range(n_samples):
        channel1 = adc.read(4) if ch1 else none_array1
        datos1.append(channel1)
        datos1.append(int.from_bytes(channel1, byteorder='big', signed=True))

    print("{} = ({})".format('CH0', datos0)) if ch0 else None
    print("{} = ({})".format('CH1', datos1)) if ch1 else None

    adc.close()


if __name__ == '__main__':
    main()
