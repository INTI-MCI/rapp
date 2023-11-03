import logging
import argparse

from rapp import analysis
from rapp import polarimeter


HELP_CYCLES = 'n° of cycles to run.'
HELP_STEP = 'every how many degrees to take a measurement.'
HELP_SAMPLES = 'n° of samples per angle.'
HELP_DELAY_POSITION = 'the delay (in seconds) after changing analyzer position.'
HELP_DELAY_ANGLE = 'the delay (in seconds) after measuring an angle.'
HELP_ANALYZER_V = 'velocity of the analyzer.'
HELP_FILENAME = 'filename in which to write results.'
HELP_TEST = 'whether to run on test mode. No real connection will be established.'

HELP_VERBOSE = 'whether to run on verbose mode.'

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def setup_logger(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT
    )

    """
    root_logger = logging.getLogger()

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root_logger.addHandler(handler)
    """


def add_polarimeter_subparser(subparsers):
    p = subparsers.add_parser("polarimeter")
    p.add_argument('--cycles', type=int, default=1, metavar='', help=HELP_CYCLES)
    p.add_argument('--step', type=float, default=30, metavar='', help=HELP_STEP)
    p.add_argument('--samples', type=int, default=1, metavar='', help=HELP_SAMPLES)
    p.add_argument('--delay_position', type=float, default=1, metavar='', help=HELP_DELAY_POSITION)
    p.add_argument('--delay_angle', type=float, default=0, metavar='', help=HELP_DELAY_ANGLE)
    p.add_argument('--analyzer_velocity', type=float, default=4, metavar='', help=HELP_ANALYZER_V)
    p.add_argument('--filename', type=str, default='test', metavar='', help=HELP_FILENAME)
    p.add_argument('--test', action='store_true', help=HELP_TEST)
    p.add_argument('-v', '--verbose', action='store_true', help=HELP_VERBOSE)


def main():
    parser = argparse.ArgumentParser(
        prog='RAPP',
        description='Tools for measuring the rotation angle of the plane of polarization (RAPP).',
        epilog='Text at the bottom of help'
    )

    subparsers = parser.add_subparsers(dest='command', help='available commands')
    subparsers.add_parser("analysis")
    add_polarimeter_subparser(subparsers)

    args = parser.parse_args()

    setup_logger(args.verbose)

    if args.command == 'analysis':
        analysis.main()

    if args.command == 'polarimeter':
        polarimeter.main(
            cycles=args.cycles,
            step=args.step,
            samples=args.samples,
            delay_position=args.delay_position,
            delay_angle=args.delay_angle,
            analyzer_velocity=args.analyzer_velocity,
            filename=args.filename,
            test=args.test
        )


if __name__ == '__main__':
    main()
