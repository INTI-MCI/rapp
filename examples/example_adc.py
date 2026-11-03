import sys
import logging

from rapp.adc import ADC
from rapp.utils import timing

ADC_WIN_DEVICE = 'COM3'
ADC_LINUX_DEVICE = '/dev/ttyACM0'
ADC_BAUDRATE = 57600
ADC_TIMEOUT = 1  # seconds
ADC_MULTIPLIER_mV = 0.125
ADC_USED = "ADS_1115"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@timing
def handshake(adc):
    # time.sleep(5)
    # adc.write(bytes(str(0), 'utf-8'))
    # adc.write(bytes(str(0), 'utf-8'))
    for i in range(50):
        data = adc.readline().decode().strip()
        logger.info(data)


@timing
def acquire(adc: ADC, **kwargs):
    return adc.acquire(**kwargs)


def resolve_adc_device():
    if sys.platform == 'linux':
        return ADC_LINUX_DEVICE

    return ADC_WIN_DEVICE


def continuous_run(channel, n_samples, dt_mus):
    print("Instantiating ADC...")
    adc = ADC.build(resolve_adc_device(), baudrate=ADC_BAUDRATE, timeout=ADC_TIMEOUT, adc_used=ADC_USED,
                    ch0=True, ch1=True)


    print("Acquiring...")
    mean_data = adc.print_read_n_every_dt(channel, n_samples, dt_mus)
    logger.info("Mean time between samples: {} us".format(mean_data))

    adc.close()


def single_run(n_samples, ch0, ch1):
    print("Instantiating ADC...")
    adc = ADC.build(resolve_adc_device(), baudrate=ADC_BAUDRATE, timeout=ADC_TIMEOUT, adc_used=ADC_USED,
                    ch0=ch0, ch1=ch1)

    print("Acquiring...")
    data, elapsed_time = acquire(adc, n_samples=n_samples)
    sps = n_samples / elapsed_time

    channels_names_tuple = ('CH0', 'CH1')

    for d in data:
        logger.info("{} = {}".format(channels_names_tuple, d))

    logger.info("Samples per second: {}".format(sps))
    logger.info("Number of measurements: {}".format(len(data)))

    adc.close()


if __name__ == '__main__':
    try:
        if sys.argv[1] == 'single_run':
            n_samples = int(sys.argv[2])
            ch0 = bool(int(sys.argv[3]))
            ch1 = bool(int(sys.argv[4]))
            single_run(n_samples, ch0, ch1)
        elif sys.argv[1] == 'continuous_run':
            channel = int(sys.argv[2])
            n_samples = int(sys.argv[3])
            dt_mus = int(sys.argv[4])
            continuous_run(channel, n_samples, dt_mus)
    except IndexError as e:
        print(e)
        print("Need to pass input parameters are single_run SAMPLES, CH0, CH1; continuous_run CHANNEL, SAMPLES, DT_MUS")
        exit(1)
