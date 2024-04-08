import logging
from rapp.polarimeter import main as main_polarimeter

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
    e2 = dict(cycles=1, step=1, samples=169, reps=100, delay_position=1)
    e3 = dict(cycles=1, step=1, samples=1014, reps=100)
    experimentos = [e1, e2, e3]

    for i, exp in enumerate(experimentos, 1):
        logger.info("EXPERIMENTO {}: {}".format(i, exp))
        main_polarimeter(prefix='min-setup-{}'.format(i), **exp)


if __name__ == '__main__':
    main()
