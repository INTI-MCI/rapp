import time
import logging

from rapp.polarimeter import main as main_polarimeter
from rapp.motion_controller import ESP301Error


LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)


def setup_logger():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    # Hide logging from external libraries
    external_libs = ['matplotlib', 'PIL']
    for lib in external_libs:
        logging.getLogger(lib).setLevel(logging.ERROR)


def main():
    setup_logger()
    e1 = dict(cycles=1, step=1, samples=169, reps=100)
    e3 = dict(cycles=0, samples=6000000, chunk_size=600000, reps=1, no_ch1=True)
    # e2 = dict(cycles=1, step=1, samples=169, reps=100, delay_position=1)
    experimentos = [e3, e1]

    for i, exp in enumerate(experimentos, 1):
        logger.info("EXPERIMENTO {}: {}".format(i, exp))
        try:
            main_polarimeter(prefix='min-setup-detector-obj-{}'.format(i), **exp)
        except ESP301Error as e:
            logger.warning(f"Found error: {e}. Retrying...")
            time.sleep(30)
            main_polarimeter(prefix='min-setup-detector-obj-{}'.format(i), **exp)


if __name__ == '__main__':
    main()
