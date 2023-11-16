import sys
import time
import serial

from rapp.utils import timing

# ADC_DEVICE = 'COM4'
ADC_DEVICE = '/dev/ttyACM0'

ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_MULTIPLIER_mV = 0.125

WAIT_TIME_AFTER_CONNECTION = 2


def bits_to_volts(value):
    return value * ADC_MULTIPLIER_mV / 1000


def read_data(adc, n_samples, in_bytes=True):
    data = []
    for _ in range(n_samples):
        try:
            if in_bytes:
                value = adc.read(2)
                value = int.from_bytes(value, byteorder='big', signed=True)
            else:
                value = adc.readline().decode().strip()
                value = int(value)

            value = bits_to_volts(value)
            data.append(value)
        except (ValueError, UnicodeDecodeError) as e:
            print(e)

    return data


def acquire(adc, n_samples, **kwargs):
    adc.write(bytes(str(n_samples), 'utf-8'))
    # Sending directly the numerical value didn't work.
    # See: https://stackoverflow.com/questions/69317581/sending-serial-data-to-arduino-works-in-serial-monitor-but-not-in-python  # noqa

    a0 = read_data(adc, n_samples, **kwargs)
    a1 = read_data(adc, n_samples, **kwargs)

    data = zip(a0, a1)

    return data


def main(n_samples=5):
    adc = serial.Serial(ADC_DEVICE, ADC_BAUDRATE, timeout=ADC_TIMEOUT)

    print("Waiting {} seconds after opening the connection...".format(WAIT_TIME_AFTER_CONNECTION))
    # Arduino resets when a new serial connection is made.
    # We need to wait, otherwise we don't recieve anything.
    # Less than 2 seconds does not work.
    # TODO: check if we can avoid that arduino resets.
    time.sleep(WAIT_TIME_AFTER_CONNECTION)

    print("Flushing input...")
    adc.flushInput()

    print("Acquiring...")
    data, elapsed_time = timing(acquire)(adc, n_samples=n_samples, in_bytes=False)

    for d in data:
        print("(A0, A1) = {}".format(d))

    sps = n_samples / elapsed_time
    print("Samples per second: {}".format(sps))

    adc.close()


if __name__ == '__main__':
    try:
        n_samples = int(sys.argv[1])
    except ValueError as e:
        print(e)
        print("Input parameter must be a integer number!")
        exit(1)

    main(n_samples)
