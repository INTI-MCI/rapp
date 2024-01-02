import sys

from rapp.utils import timing
from rapp.adc import ADC


ADC_WIN_DEVICE = 'COM4'
ADC_LINUX_DEVICE = '/dev/ttyACM0'
ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_MULTIPLIER_mV = 0.125
ADC_WAIT_TIME = 5


@timing
def handshake(adc):
    # time.sleep(5)
    # adc.write(bytes(str(0), 'utf-8'))
    # adc.write(bytes(str(0), 'utf-8'))
    for i in range(50):
        data = adc.readline().decode().strip()
        print(data)


@timing
def acquire(adc, **kwargs):
    return adc.acquire(**kwargs)


def resolve_adc_device():
    if sys.platform == 'linux':
        return ADC_LINUX_DEVICE

    return ADC_WIN_DEVICE


def main(n_samples, ch0, ch1):
    print("Instantiating ADC...")
    adc = ADC(resolve_adc_device(), b=ADC_BAUDRATE, timeout=ADC_TIMEOUT, wait=ADC_WAIT_TIME)

    print("Acquiring...")
    data, elapsed_time = acquire(adc, n_samples=n_samples, ch0=ch0, ch1=ch1)
    sps = n_samples / elapsed_time

    channels_names_tuple = "({})".format(", ".join(data.keys()))

    values = list(zip(*data.values()))

    for d in values:
        print("{} = {}".format(channels_names_tuple, d))

    print("Samples per second: {}".format(sps))
    print("Number of measurements: {}".format(len(data)))

    adc.close()


if __name__ == '__main__':
    try:
        n_samples = int(sys.argv[1])
        ch0 = bool(int(sys.argv[2]))
        ch1 = bool(int(sys.argv[3]))
    except IndexError as e:
        print(e)
        print("Need to pass input parameters are SAMPLES, CH0, CH1")
        exit(1)

    main(n_samples, ch0, ch1)
