import sys

from rapp.utils import timing
from rapp.adc import ADC


# ADC_DEVICE = 'COM4'
ADC_DEVICE = '/dev/ttyACM0'


def acquire(adc, *args, **kwargs):
    return adc.acquire(*args, **kwargs)


def main(n_samples=5):
    print("Initializing ADC...")
    adc = ADC(ADC_DEVICE, b=57600, timeout=0.1)

    print("Flushing input...")
    adc.flush_input()

    print("Acquiring...")
    data, elapsed_time = timing(acquire)(adc, n_samples=n_samples, in_bytes=True)

    for d in data:
        print("(A0, A1) = {}".format(d))

    sps = n_samples / elapsed_time
    print("Samples per second: {}".format(sps))

    adc.close()


if __name__ == '__main__':
    try:
        n_samples = int(sys.argv[1])
    except IndexError:
        print("ERROR: You must provide an argument (number of samples). ")
        exit(1)
    except ValueError:
        print("ERROR: Number of samples should be an integer number!")
        exit(1)

    main(n_samples)
