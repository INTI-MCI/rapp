import sys
import time
import serial
import numpy as np
from matplotlib import pyplot as plt

from rapp.utils import timing
from rapp.adc import ADC

from rich.progress import track

'''
Así como quedó escrita puede reemplazar el serial monitor del Arduino IDE para chequear el voltaje
que sale de los fotodiodos, modificando los parámetros de la funcion main según sea necesario
'''

ADC_WIN_DEVICE = 'COM3'
ADC_LINUX_DEVICE = '/dev/ttyACM0'
ADC_BAUDRATE = 57600
ADC_TIMEOUT = 2
ADC_TIMEOUT_OPEN = 7

ADC_MULTIPLIER_mV = 0.125
ADC_WAIT_TIME = 5

CMD_TEMPLATE = "{measurement};{ch0};{ch1};{samples};\n"
AVAILABLE_CHANNELS = ['CH0', 'CH1']

PLOT_CHANNELS = True

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
    connection = True
    while elapsed_time < ADC_TIMEOUT_OPEN:
        self.write(b'ready?\n')
        output = self.readline()
        end = time.time()
        elapsed_time = end - start
        # print('output: ', output)
        if output == b'yes\r\n':
            line1 = self.readline()
            # self.reset_input_buffer()
            # print("Line 1 left in input buffer: {}".format(line1))
            # line2 = self.readline()
            # print("Line 2 left in input buffer: {}".format(line2))
            # if self.in_waiting():
            #     print("Connection opened, buffer: {}".format(self.inWaiting()))
            #     self.reset_input_buffer()
            #     print("Connection opened, buffer: {}".format(self.inWaiting()))
            #     buf = self.read(self.in_waiting())
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
        connection = False
    print(
        "Connection opened in {:.2f} seconds with {} queries.".format(elapsed_time, queries)
    )
    return connection


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
        self.reset_input_buffer()

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

def bits_to_volts(value):
        return value * (1240/2**23) / 1000

def read_times(serial_connection):
    times_ino = [] # Times que se miden a nivel del .ino
    for i in range(5):
        times_byte = serial_connection.read(4)
        times_int = int.from_bytes(times_byte, byteorder='little', signed=False)
        times_ino.append(times_int)
    return times_ino

def read_profiler(self, ch0=0, ch1=0):
    profiler0 = []  # [elapsed_DRDY, max_DRDY, min_DRDY, elapsed_read, max_read, min_read]
    profiler1 = []
    if ch0:
        for i in range(6):
            profiled_time = self.read(4)
            prof_time_int = int.from_bytes(profiled_time, byteorder='little', signed=False)
            profiler0.append(prof_time_int)
    if ch1:
        for i in range(6):
            profiled_time = self.read(4)
            prof_time_int = int.from_bytes(profiled_time, byteorder='little', signed=False)
            profiler1.append(prof_time_int)

    return profiler0, profiler1


def main(n_samples=300, ch0=1, ch1=1, total_samples=1, profile=False):
    # n_samples: cantidad de muestras que se piden
    # ch0 y ch1 dicen si se pide o no ese canal
    # total_samples: cantidad de veces que se piden las n_samples 
    # Si profile es 1, pide e imprime los datos del profiler

    print("Asking for {} samples from channels CH0={} and CH1={} for {} times.".format(n_samples, ch0, ch1, total_samples))
    print("Instantiating ADC...")
    adc = get_serial_connection(ADC_WIN_DEVICE, baudrate=ADC_BAUDRATE, timeout=ADC_TIMEOUT)
    # adc = ADC(resolve_adc_device(), timeout_open=ADC_TIMEOUT_OPEN)#, baudrate=ADC_BAUDRATE, timeout=ADC_TIMEOUT)
    connection = wait_for_connection(adc)
    if not connection:
        adc.close()
        return
    else:
        # print("Connection opened.")
        # buf = adc.readline()
        # print(buf)
        adc.reset_input_buffer()
        # time.sleep(5)
        # buf2 = adc.readline()
        # print(buf2)
        for j in range(total_samples):
            adc.write(bytes(CMD_TEMPLATE.format(measurement='adc', ch0=ch0, ch1=ch1, samples=n_samples).encode('utf-8')))
            # adc.write(bytes('adc_n_dt;{};500;\n'.format(n_samples).encode('utf-8')))
            none_array0 = np.full(n_samples, None)
            none_array1 = np.full(n_samples, None)
            datos0 = []
            datos1 = []

            for i in range(n_samples):
                channel0 = adc.read(4) if ch0 else none_array0
                datos0.append(bits_to_volts(int.from_bytes(channel0, byteorder='big', signed=True)))

            for i in range(n_samples):
                channel1 = adc.read(4) if ch1 else none_array1
                datos1.append(bits_to_volts(int.from_bytes(channel1, byteorder='big', signed=True)))

            mean0 = np.mean(datos0) if ch0 else None
            mean1 = np.mean(datos1) if ch1 else None
            print("{}. Media = {:.4e}. Std = {:.2e}. MinDif = {:.2e}. MaxDif = {:.2e}".format('CH0', np.mean(datos0), np.std(datos0), np.min(datos0) - mean0, np.max(datos0) - mean0)) if ch0 else None
            print("{}. Media = {:.4e}. Std = {:.2e}. MinDif = {:.2e}. MaxDif = {:.2e}".format('CH1', np.mean(datos1), np.std(datos1), np.min(datos1) - mean1, np.max(datos1) - mean1)) if ch1 else None

            if PLOT_CHANNELS:
                fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                axs[0, 0].plot(datos0, color='blue')
                axs[0, 0].set_title('Channel 0 Voltage vs Sample Index')
                axs[0, 0].set_xlabel('Sample Index')
                axs[0, 0].set_ylabel('Voltage (V)')
                axs[1, 0].hist(datos0, bins=50, color='blue', alpha=0.7)
                axs[1, 0].set_title('Channel 0 Histogram')
                axs[1, 0].set_xlabel('Voltage (V)')
                axs[1, 0].set_ylabel('Frequency')
                axs[0, 1].plot(datos1, color='red')
                axs[0, 1].set_title('Channel 1 Voltage vs Sample Index')
                axs[0, 1].set_xlabel('Sample Index')
                axs[0, 1].set_ylabel('Voltage (V)')
                axs[1, 1].hist(datos1, bins=50, color='red', alpha=0.7)
                axs[1, 1].set_title('Channel 1 Histogram')
                axs[1, 1].set_xlabel('Voltage (V)')
                axs[1, 1].set_ylabel('Frequency')
                plt.show()

            adc.write(bytes("measured_times?\n", 'utf-8'))
            times = read_times(adc)

            print('Channel 0 time = {} us. Sample rate = {:.0f} sps'.format(times[1], n_samples / (times[1] / 1e6)))
            print('Channel 1 time = {} us. Sample rate = {:.0f} sps'.format(times[2], n_samples / (times[2] / 1e6)))
            print('Send channel 0 data time per sample = {:.2f} us'.format(times[3] / n_samples))
            print('Send channel 1 data time per sample = {:.2f} us'.format(times[4] / n_samples))
            print('Total = {} us'.format(times[0]))

            if profile:
                adc.write(bytes("profiled_times?\n", 'utf-8'))
                
                profiler0, profiler1 = read_profiler(adc, ch0, ch1)

                # profiler0 es [elapsed_DRDY, max_DRDY, min_DRDY, elapsed_read, max_read, min_read]
                if ch0:
                    print('Profiler ch0 / us')
                    print('DRDY: elapsed = {}, Min = {}, Max = {}'.format(profiler0[0], profiler0[2], profiler0[1]))
                    print('read: elapsed = {}, Min = {}, Max = {}'.format(profiler0[3], profiler0[5], profiler0[4]))
                if ch1:
                    print('Profiler ch1 / us')
                    print('DRDY: elapsed = {}, Min = {}, Max = {}'.format(profiler1[0], profiler1[2], profiler1[1]))
                    print('read: elapsed = {}, Min = {}, Max = {}'.format(profiler1[3], profiler1[5], profiler1[4]))

            print('--------------')

            time.sleep(0.2)

        adc.close()


if __name__ == '__main__':
    main()
