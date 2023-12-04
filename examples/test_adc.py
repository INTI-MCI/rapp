import sys

from rapp.utils import timing
from rapp.adc import ADC


ADC_DEVICE = 'COM4'
# ADC_DEVICE = '/dev/ttyACM0'

ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_MULTIPLIER_mV = 0.125

WAIT_TIME_AFTER_CONNECTION = 5


def bits_to_volts(value):
    return value * ADC_MULTIPLIER_mV / 1000


@timing
def read_line(adc):
    return adc.readline().decode().strip()


@timing
def read_data(adc, n_samples):
    data = []
    for _ in range(n_samples):
        try:
            value, _ = read_line(adc)
            if value:
                value = int(value)
                value = bits_to_volts(value)
                data.append(value)
        except (ValueError, UnicodeDecodeError) as e:
            print(e)

    return data


@timing
def handshake(adc):
    # time.sleep(5)
    # adc.write(bytes(str(0), 'utf-8'))
    # adc.write(bytes(str(0), 'utf-8'))
    for i in range(50):
        data = adc.readline().decode().strip()
        print(data)


@timing
def acquire(adc, n_samples):
    return adc.acquire(n_samples)


def main(n_samples=5):

    adc = ADC(ADC_DEVICE, b=ADC_BAUDRATE, timeout=ADC_TIMEOUT, wait=WAIT_TIME_AFTER_CONNECTION)

    print("Acquiring...")
    data, elapsed_time = acquire(adc, n_samples=n_samples)

    for d in data:
        print("(A0, A1) = {}".format(d))

    sps = n_samples / elapsed_time
    print("Samples per second: {}".format(sps))
    print("Number of measurements: {}".format(len(data)))

    adc.close()


if __name__ == '__main__':
    try:
        n_samples = int(sys.argv[1])
    except ValueError as e:
        print(e)
        print("Input parameter must be a integer number!")
        exit(1)

    main(n_samples)
