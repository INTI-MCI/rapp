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
    e1 = dict(cycles=0, samples=760500, chunk_size=169, reps=1)
    e2 = dict(cycles=0, samples=760500, chunk_size=1521000, reps=1)
    e3 = dict(cycles=1.5, step=0.5, samples=338, reps=50)
    e4 = dict(cycles=1, step=1, samples=169, reps=50, velocity=0.5)
    e5 = dict(cycles=1, step=1, samples=169, reps=50, velocity=1)
    e6 = dict(cycles=1, step=1, samples=169, reps=50, velocity=2)
    experimentos = [e1, e2]
    # time.sleep(5400)
    basename = "noise-quartz" 
    for i, exp in enumerate(experimentos, 1):
        logger.info("EXPERIMENTO {}: {}".format(i, exp))
        prefix = f"{basename}-{i}"
        try:
            for k, v in exp.items():
                prefix + f"{k}-{v}"
            main_polarimeter(prefix=prefix, **exp)
        except ESP301Error as e:
            logger.warning(f"Found error: {e}. Retrying...")
            time.sleep(30)
            main_polarimeter(prefix=prefix, **exp)


if __name__ == '__main__':
    main()
