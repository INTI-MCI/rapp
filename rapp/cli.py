import sys
import argparse

from rapp import polarimeter
from rapp.signal import analysis
from rapp.signal import simulator
from rapp.log import setup_logger

HELP_POLARIMETER = "Tool for measuring signals with the polarimeter."
HELP_SIM = "Tool for making numerical simulations."
HELP_PHASE_DIFF = 'Tool for calculating phase difference between two harmonic signals.'
HELP_ANALYSYS = "Tool for analyzing signals: noise, drift, etc."

HELP_CYCLES = 'n° of cycles to run.'
HELP_STEP = 'every how many degrees to take a measurement.'
HELP_SAMPLES = 'n° of samples per angle.'
HELP_DELAY_POSITION = 'delay (in seconds) after changing analyzer position (default: %(default)s).'
HELP_VELOCITY = 'velocity of the analyzer in deg/s (default: %(default)s).'
HELP_PREFIX = 'prefix for the filename in which to write results (default: %(default)s).'
HELP_TEST = 'run on test mode. No real connections will be established (default: %(default)s).'
HELP_PLOT = 'plot the results when the measurement is finished (default: %(default)s).'
HELP_VERBOSE = 'whether to run with DEBUG log level (default: %(default)s).'
HELP_METHOD = 'phase difference calculation method (default: %(default)s).'

HELP_SIM_REPS = 'number of repetitions in each simulated iteration (default: %(default)s).'

HELP_SHOW = 'whether to show the plot.'
HELP_FILEPATH = 'the file containing the measurements.'

HELP_SIM_NAME = (
    'name of the simulation. '
    'One of {}.'.format(simulator.SIMULATIONS)
)

EXAMPLE_POLARIMETER = "rapp polarimeter --cycles 1 --step 30 --samples 10 --delay_position 0"
EPILOG_POLARIMETER = "Example: {}".format(EXAMPLE_POLARIMETER)

EXAMPLE_SIM = "rapp sim error_vs_cycles --show"
EPILOG_SIM = "Example: {}".format(EXAMPLE_SIM)

EXAMPLE_PHASE_DIFF = "rapp phase_diff data/test-cycles2-step1.0-samples50.txt"
EPILOG_PHASE_DIFF = "Example: {}".format(EXAMPLE_PHASE_DIFF)


def add_polarimeter_subparser(subparsers):
    p = subparsers.add_parser("polarimeter", help=HELP_POLARIMETER, epilog=EPILOG_POLARIMETER)
    p.add_argument('--cycles', type=int, required=True, help=HELP_CYCLES)
    p.add_argument('--step', type=float, default=10, help=HELP_STEP)
    p.add_argument('--samples', type=int, required=True, help=HELP_SAMPLES)
    p.add_argument('--chunks', type=int)
    p.add_argument('--delay_position', type=float, default=1, metavar='', help=HELP_DELAY_POSITION)
    p.add_argument('--velocity', type=float, default=4, metavar='', help=HELP_VELOCITY)
    p.add_argument('--prefix', type=str, default='test', metavar='', help=HELP_PREFIX)
    p.add_argument('--test', action='store_true', help=HELP_TEST)
    p.add_argument('--plot', action='store_true', help=HELP_PLOT)
    p.add_argument('-v', '--verbose', action='store_true', help=HELP_VERBOSE)


def add_sim_subparser(subparsers):
    p = subparsers.add_parser("sim", help=HELP_SIM, epilog=EPILOG_SIM)
    p.add_argument('name', type=str, help=HELP_SIM_NAME)
    p.add_argument('--samples', type=int, default=1, help=HELP_SAMPLES)
    p.add_argument('--reps', type=int, default=1, help=HELP_SIM_REPS)
    p.add_argument('--show', action='store_true', help=HELP_SHOW)
    p.add_argument('-v', '--verbose', action='store_true', help=HELP_VERBOSE)


def add_phase_diff_subparser(subparsers):
    p = subparsers.add_parser("phase_diff", help=HELP_PHASE_DIFF, epilog=EPILOG_PHASE_DIFF)
    p.add_argument('filepath', type=str, help=HELP_FILEPATH)
    p.add_argument('--method', type=str, default='odr', help=HELP_METHOD)
    p.add_argument('--show', action='store_true', help=HELP_SHOW)
    p.add_argument('-v', '--verbose', action='store_true', help=HELP_VERBOSE)


def add_analysis_subparser(subparsers):
    p = subparsers.add_parser("analysis", help=HELP_ANALYSYS)
    p.add_argument('--show', action='store_true', help=HELP_SHOW)
    p.add_argument('-v', '--verbose', action='store_true', help=HELP_VERBOSE)


def main():
    parser = argparse.ArgumentParser(
        prog='RAPP',
        description='Tools for measuring the rotation angle of the plane of polarization (RAPP).',
        # epilog='Text at the bottom of help'
    )

    subparsers = parser.add_subparsers(dest='command', help='available commands')

    add_polarimeter_subparser(subparsers)
    add_phase_diff_subparser(subparsers)
    add_analysis_subparser(subparsers)
    add_sim_subparser(subparsers)

    args = parser.parse_args(args=sys.argv[1:] or ['--help'])

    try:
        if args.command == 'phase_diff':
            setup_logger(args.verbose)
            analysis.plot_phase_difference(args.filepath, method=args.method, show=args.show)

        if args.command == 'analysis':
            setup_logger(args.verbose)
            analysis.main(show=args.show)

        if args.command == 'sim':
            setup_logger(args.verbose)
            simulator.main(args.name, reps=args.reps, samples=args.samples, show=args.show)

        if args.command == 'polarimeter':
            setup_logger(args.verbose)
            polarimeter.main(
                cycles=args.cycles,
                step=args.step,
                samples=args.samples,
                delay_position=args.delay_position,
                velocity=args.velocity,
                prefix=args.prefix,
                test=args.test,
                plot=args.plot
            )
    except ValueError as e:
        print("ERROR: {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
