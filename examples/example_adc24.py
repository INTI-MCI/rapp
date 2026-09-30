import sys
import time
import serial
import numpy as np

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
        "Connection opened in {} seconds with {} queries.".format(elapsed_time, queries)
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

def read_profiler(self, ch0=0, ch1=0):
    times_ino = [] # Times que se miden a nivel del .ino
    for i in range(3):
        times_byte = self.read(4)
        times_int = int.from_bytes(times_byte, byteorder='big', signed=False)
        times_ino.append(times_int)

    profiler0 = []  # [elapsed_DRDY, min_DRDY, max_DRDY, elapsed_read, min_read, max_read]
    profiler1 = []
    if ch0:
        for i in range(6):
            profiled_time = self.read(4)
            prof_time_int = int.from_bytes(profiled_time, byteorder='big', signed=False)
            profiler0.append(prof_time_int)
    if ch1:
        for i in range(6):
            profiled_time = self.read(4)
            prof_time_int = int.from_bytes(profiled_time, byteorder='big', signed=False)
            profiler1.append(prof_time_int)

    return times_ino, profiler0, profiler1


def main(n_samples=1, ch0=1, ch1=1, total_samples=2, profile=1):
    # n_samples: cantidad de muestras que se piden
    # ch0 y ch1 dicen si se pide o no ese canal
    # total_samples: cantidad de veces que se piden las n_samples 
    # Si profile es 1, pide e imprime los datos del profiler

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
                # channel0 = adc.readline()
                # datos0.append(channel0)
                datos0.append(bits_to_volts(int.from_bytes(channel0, byteorder='big', signed=True)))

            for i in range(n_samples):
                channel1 = adc.read(4) if ch1 else none_array1
                # channel1 = adc.readline()
                # datos1.append(channel1)
                datos1.append(bits_to_volts(int.from_bytes(channel1, byteorder='big', signed=True)))

            print("{} = ({})".format('CH0', datos0), end=' ') if ch0 else None
            print("{} = ({})".format('CH1', datos1)) if ch1 else None

            if profile:
                adc.write(bytes("profiled_times?\n", 'utf-8'))
                
                times, profiler0, profiler1 = read_profiler(adc, ch0, ch1)

                # print('times[1] = {}'.format(times[1]))
                # print('times[2] = {}'.format(times[2]))
                # print('times[0] = {}'.format(times[0]))
                
                # print('Suma de times[1] y times[2] = {}'.format(times[1] + times[2]))
                
                diferencia = times[0]-(times[1] + times[2])
                print('Diferencia entre times[0] y la suma = {}'.format(diferencia))

                # La suma de times 1 y 2 no siempre da como resultado times 0
                # Da algunos valores que parecen repetirse: 0, -4294901760, -671088668, -1174405148,
                # -1815281664, -3487694848, -469893120, -3633447454, -3089104896

                # profiler0 es [elapsed_DRDY, min_DRDY, max_DRDY, elapsed_read, min_read, max_read]
                if ch0:
                    print('Profiler ch0')
                    print('DRDY: elapsed = {}, Min = {}, Max = {}'.format(profiler0[0], profiler0[1], profiler0[2]))
                    print('read: elapsed = {}, Min = {}, Max = {}'.format(profiler0[3], profiler0[4], profiler0[5]))
                if ch1:
                    print('Profiler ch1')
                    print('DRDY: elapsed = {}, Min = {}, Max = {}'.format(profiler1[0], profiler1[1], profiler1[2]))
                    print('read: elapsed = {}, Min = {}, Max = {}'.format(profiler1[3], profiler1[4], profiler1[5]))
                # A veces el mínimo da mas grande que el máximo (??)

                print('--------------')

                # adc.reset_input_buffer() # Probé si con esto mejoraba pero parece ser lo mismo

            time.sleep(0.2)

        adc.close()


if __name__ == '__main__':
    main()
