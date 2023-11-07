import sys
import logging
import argparse

from rapp import polarimeter
from rapp.signal import analysis

HELP_CYCLES = 'n° of cycles to run.'
HELP_STEP = 'every how many degrees to take a measurement.'
HELP_SAMPLES = 'n° of samples per angle.'
HELP_DELAY_POSITION = 'delay (in seconds) after changing analyzer position (default: %(default)s).'
HELP_ANALYZER_V = 'velocity of the analyzer (default: %(default)s).'
HELP_PREFIX = 'prefix for the filename in which to write results (default: %(default)s).'
HELP_TEST = 'run on test mode. No real connections will be established (default: %(default)s).'

HELP_VERBOSE = 'whether to run with DEBUG log level (default: %(default)s).'

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def setup_logger(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT
    )


def add_polarimeter_subparser(subparsers):
    p = subparsers.add_parser("polarimeter")
    p.add_argument('--cycles', type=int, required=True, help=HELP_CYCLES)
    p.add_argument('--step', type=float, required=True, help=HELP_STEP)
    p.add_argument('--samples', type=int, required=True, help=HELP_SAMPLES)
    p.add_argument('--delay_position', type=float, default=1, metavar='', help=HELP_DELAY_POSITION)
    p.add_argument('--analyzer_velocity', type=float, default=4, metavar='', help=HELP_ANALYZER_V)
    p.add_argument('--prefix', type=str, default='test', metavar='', help=HELP_PREFIX)
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

    args = parser.parse_args(args=sys.argv[1:] or ['--help'])

    if args.command == 'analysis':
        analysis.main()

    if args.command == 'polarimeter':
        setup_logger(args.verbose)
        polarimeter.main(
            cycles=args.cycles,
            step=args.step,
            samples=args.samples,
            delay_position=args.delay_position,
            analyzer_velocity=args.analyzer_velocity,
            prefix=args.prefix,
            test=args.test
        )


if __name__ == '__main__':
    main()
