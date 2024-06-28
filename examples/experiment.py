import time
import logging

from rapp.polarimeter import run as main_polarimeter
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
    e1 = dict(cycles=1, step=1, samples=169, reps=50, velocity=1, acceleration=0.5, deceleration=0.5)
    e2 = dict(cycles=1, step=1, samples=169, reps=50, acceleration=4)
    e3 = dict(cycles=1, step=1, samples=169, reps=50, acceleration=12)
    e4 = dict(cycles=1, step=1, samples=169, reps=50, velocity=0.5)
    e5 = dict(cycles=1, step=1, samples=169, reps=50, velocity=1)
    e6 = dict(cycles=1, step=1, samples=169, reps=50, velocity=2)
    experimentos = [e1]

    time.sleep(7200)
    name = "quartz-vel-1-acc-0.5-deacc-0.5" 
    for i, exp in enumerate(experimentos, 1):
        logger.info("EXPERIMENTO {}: {}".format(i, exp))
        try:
            main_polarimeter(prefix='{}-{}'.format(name, i), **exp)
        except ESP301Error as e:
            logger.warning(f"Found error: {e}. Retrying...")
            time.sleep(30)
            main_polarimeter(prefix='{}-{}'.format(name, i), **exp)


if __name__ == '__main__':
    main()
