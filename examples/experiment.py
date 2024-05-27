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
    e1 = dict(cycles=1, step=1, samples=169, reps=50)
    e2 = dict(cycles=2, step=1, samples=169, reps=50)
    e3 = dict(cycles=1, step=0.5, samples=169, reps=50)
    e4 = dict(cycles=1, step=1, samples=1218, reps=50)
    e5 = dict(cycles=2, step=2, samples=1218, reps=50)
    experimentos = [e1, e2, e3, e4, e5]

    for i, exp in enumerate(experimentos, 1):
        logger.info("EXPERIMENTO {}: {}".format(i, exp))
        try:
            main_polarimeter(prefix='full-setup-norm-{}'.format(i), **exp)
        except ESP301Error as e:
            logger.warning(f"Found error: {e}. Retrying...")
            time.sleep(30)
            main_polarimeter(prefix='full-setup-norm-{}'.format(i), **exp)


if __name__ == '__main__':
    main()
