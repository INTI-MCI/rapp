import logging
import sys
import time

from rapp.adc import ADC
from rapp.utils import timing

ADC_WIN_DEVICE = 'COM3'
ADC_LINUX_DEVICE = '/dev/ttyACM0'
ADC_BAUDRATE = 57600
ADC_TIMEOUT = 1  # seconds
ADC_MULTIPLIER_mV = 0.125
ADC_USED = "ADS_1232"

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
    logger.info("Instantiating ADC...")
    adc = ADC.build(resolve_adc_device(), baudrate=ADC_BAUDRATE, timeout=ADC_TIMEOUT, adc_used=ADC_USED,
                    ch0=True, ch1=True)


    logger.info("Acquiring...")
    mean_data = adc.print_read_n_every_dt(channel, n_samples, dt_mus)
    logger.info("Mean time between samples: {} us".format(mean_data))

    adc.close()


def single_run(n_samples, ch0, ch1, n_runs=1, delay_between_runs=0.0):
    EN_HEXA = False
    channels_names_tuple = ('CH0', 'CH1')
    logger.info("Instantiating ADC...")
    adc = ADC.build(resolve_adc_device(), baudrate=ADC_BAUDRATE, timeout=ADC_TIMEOUT, adc_used=ADC_USED,
                    ch0=ch0, ch1=ch1)

    logger.info("Self calibrating...")
    adc.self_calibrate()
    logger.info("Acquiring...")
    for i in range(n_runs):
        data, elapsed_time = acquire(adc, n_samples=n_samples, hex=EN_HEXA)

        if data is None:
            logger.error("No data acquired")
            adc.close()
            return
        for d in data:
            if EN_HEXA:
                d = [f"0x{a_d & 0xFFFFFFFF:08X}" for a_d in d]
            logger.info("{} = {}".format(channels_names_tuple, d))
        time.sleep(delay_between_runs)

    sps = n_samples / elapsed_time * n_runs
    logger.info("Samples per second: {}".format(sps))
    logger.info("Number of measurements: {}".format(n_runs * n_samples))

    adc.close()


if __name__ == '__main__':
    try:
        if sys.argv[1] == 'single_run':
            n_samples = int(sys.argv[2])
            ch0 = bool(int(sys.argv[3]))
            ch1 = bool(int(sys.argv[4]))
            single_run(n_samples, ch0, ch1)
        elif sys.argv[1] == 'multi_run':
            n_samples = int(sys.argv[2])
            ch0 = bool(int(sys.argv[3]))
            ch1 = bool(int(sys.argv[4]))
            n_runs = int(sys.argv[5])
            single_run(n_samples, ch0, ch1, n_runs=n_runs)
        elif sys.argv[1] == 'continuous_run':
            channel = int(sys.argv[2])
            n_samples = int(sys.argv[3])
            dt_mus = int(sys.argv[4])
            continuous_run(channel, n_samples, dt_mus)
    except IndexError as e:
        logger.error(e)
        logger.error("Need to pass input parameters are single_run SAMPLES, CH0, CH1; continuous_run CHANNEL, SAMPLES, DT_MUS")
        sys.exit(1)
