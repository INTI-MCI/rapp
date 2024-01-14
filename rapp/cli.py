import sys
import argparse

from rapp import polarimeter
from rapp.analysis import analysis, optical_rotation, phase_diff, raw
from rapp.simulations import simulator
from rapp.log import setup_logger

HELP_POLARIMETER = "Tool for measuring signals with the polarimeter."
HELP_SIM = "Tool for making numerical simulations."
HELP_PHASE_DIFF = 'Tool for calculating phase difference from single polarimeter measurement.'
HELP_OR = 'Tool for calculating optical rotation from initial phase and final phase measurements.'
HELP_ANALYSYS = "Tool for analyzing signals: noise, drift, etc."
HELP_PLOT_ROW = "Tool for plotting raw measurements."

HELP_SAMPLES = 'n° of samples per angle.'
HELP_CYCLES = 'n° of cycles to run (default: %(default)s).'
HELP_STEP = 'motion step of the rotating analyzer (default: %(default)s).'
HELP_DELAY_POSITION = 'delay (in seconds) after changing analyzer position (default: %(default)s).'
HELP_VELOCITY = 'velocity of the analyzer in deg/s (default: %(default)s).'
HELP_INIT_P = 'initial position of the analyzer in deg (default: %(default)s).'
HELP_NOCH0 = 'excludes channel 0 from measurement (default: %(default)s).'
HELP_NOCH1 = 'excludes channel 1 from measurement (default: %(default)s).'
HELP_CHUNK_SIZE = 'measure data in chunks of this size. If 0, no chunks (default: %(default)s).'
HELP_PREFIX = 'prefix for the filename in which to write results (default: %(default)s).'
HELP_TEST_ESP = 'use ESP mock object. (default: %(default)s).'
HELP_TEST_ADC = 'use ADC mock object. (default: %(default)s).'
HELP_PLOT = 'plot the results when the measurement is finished (default: %(default)s).'
HELP_VERBOSE = 'whether to run with DEBUG log level (default: %(default)s).'
HELP_OVERWRITE = 'whether to overwrite existing files without asking (default: %(default)s).'
HELP_REPS = 'Number of repetitions (default: %(default)s).'

HELP_HWP_CYCLES = 'n° of cycles of the HW plate (default: %(default)s).'
HELP_HWP_STEP = 'motion step of the rotating HW plate (default: %(default)s).'
HELP_HWP_DELAY = 'delay (in seconds) after changing HW plate position (default: %(default)s).'


HELP_METHOD = 'phase difference calculation method (default: %(default)s).'
HELP_HWP = 'whether the measurement was done with a half wave plate (default: %(default)s).'

HELP_SIM_REPS = 'number of repetitions in each simulated iteration (default: %(default)s).'

HELP_SHOW = 'whether to show the plot (default: %(default)s).'
HELP_FILEPATH = 'file containing the measurements.'
HELP_FOLDER = 'folder containing the measurements.'

HELP_FOLDER_WITHOUT_SAMPLE = 'folder containing the measurements without sample.'
HELP_FOLDER_WITH_SAMPLE = 'folder containing the measurements with sample.'


HELP_SIM_NAME = (
    'name of the simulation. '
    'One of {}.'.format(simulator.SIMULATIONS)
)

HELP_ANALYSIS_NAME = (
    'name of the analysis. '
    'One of {}.'.format(analysis.ANALYSIS_NAMES)
)

EXAMPLE_POLARIMETER = "rapp polarimeter --cycles 1 --step 30 --samples 10 --delay-position 0"
EPILOG_POLARIMETER = "Example: {}".format(EXAMPLE_POLARIMETER)

EXAMPLE_SIM = "rapp sim error_vs_samples --show"
EPILOG_SIM = "Example: {}".format(EXAMPLE_SIM)

EXAMPLE_PHASE_DIFF = "rapp phase_diff data/sine-range4V-632nm-cycles2-step1.0-samples50.txt"
EPILOG_PHASE_DIFF = "Example: {}".format(EXAMPLE_PHASE_DIFF)

EXAMPLE_PLOT_RAW = "rapp plot_raw data/sine-range4V-632nm-cycles2-step1.0-samples50.txt"
EPILOG_PLOT_RAW = "Example: {}".format(EXAMPLE_PHASE_DIFF)


def add_polarimeter_subparser(subparsers):
    p = subparsers.add_parser("polarimeter", help=HELP_POLARIMETER, epilog=EPILOG_POLARIMETER)
    p.add_argument('--samples', type=int, required=True, help=HELP_SAMPLES)
    p.add_argument('--cycles', type=int, default=0, help=HELP_CYCLES)
    p.add_argument('--step', type=float, default=45, help=HELP_STEP)
    p.add_argument('--chunk-size', type=int, default=500, metavar='', help=HELP_CHUNK_SIZE)
    p.add_argument('--delay-position', type=float, default=1, metavar='', help=HELP_DELAY_POSITION)
    p.add_argument('--velocity', type=float, default=4, metavar='', help=HELP_VELOCITY)
    p.add_argument('--init-position', type=float, default=None, metavar='', help=HELP_INIT_P)
    p.add_argument('--hwp-cycles', type=int, default=0, metavar='', help=HELP_HWP_CYCLES)
    p.add_argument('--hwp-step', type=float, default=45, metavar='', help=HELP_HWP_STEP)
    p.add_argument('--hwp-delay', type=float, default=5, metavar='', help=HELP_HWP_DELAY)
    p.add_argument('--reps', type=int, default=1, metavar='', help=HELP_REPS)
    p.add_argument('--no-ch0', action='store_true', help=HELP_NOCH0)
    p.add_argument('--no-ch1', action='store_true', help=HELP_NOCH1)
    p.add_argument('--prefix', type=str, metavar='', help=HELP_PREFIX)
    p.add_argument('--mock-esp', action='store_true', help=HELP_TEST_ESP)
    p.add_argument('--mock-adc', action='store_true', help=HELP_TEST_ADC)
    p.add_argument('--plot', action='store_true', help=HELP_PLOT)
    p.add_argument('-v', '--verbose', action='store_true', help=HELP_VERBOSE)
    p.add_argument('-w', '--overwrite', action='store_true', help=HELP_OVERWRITE)


def add_sim_subparser(subparsers):
    p = subparsers.add_parser("sim", help=HELP_SIM, epilog=EPILOG_SIM)
    p.add_argument('name', type=str, help=HELP_SIM_NAME)
    p.add_argument('--method', type=str, default='ODR', help=HELP_METHOD)
    p.add_argument('--samples', type=int, default=50, help=HELP_SAMPLES)
    p.add_argument('--step', type=float, default=1, help=HELP_SAMPLES)
    p.add_argument('--reps', type=int, default=1, help=HELP_SIM_REPS)
    p.add_argument('--show', action='store_true', help=HELP_SHOW)
    p.add_argument('-v', '--verbose', action='store_true', help=HELP_VERBOSE)


def add_phase_diff_subparser(subparsers):
    p = subparsers.add_parser("phase_diff", help=HELP_PHASE_DIFF, epilog=EPILOG_PHASE_DIFF)
    p.add_argument('filepath', type=str, help=HELP_FILEPATH)
    p.add_argument('--method', type=str, default='ODR', help=HELP_METHOD)
    p.add_argument('--show', action='store_true', help=HELP_SHOW)
    p.add_argument('-v', '--verbose', action='store_true', help=HELP_VERBOSE)


def add_plot_raw_subparser(subparsers):
    p = subparsers.add_parser("plot_raw", help=HELP_PLOT_ROW, epilog=EPILOG_PLOT_RAW)
    p.add_argument('filepath', type=str, help=HELP_FILEPATH)
    p.add_argument('--no-ch0', action='store_true', help=HELP_NOCH0)
    p.add_argument('--no-ch1', action='store_true', help=HELP_NOCH1)
    p.add_argument('--show', action='store_true', help=HELP_SHOW)
    p.add_argument('-v', '--verbose', action='store_true', help=HELP_VERBOSE)


def add_or_subparser(subparsers):
    p = subparsers.add_parser("or", help=HELP_OR, epilog=EPILOG_PHASE_DIFF)
    p.add_argument('folder1', type=str, help=HELP_FOLDER_WITHOUT_SAMPLE)
    p.add_argument('folder2', type=str, help=HELP_FOLDER_WITH_SAMPLE)
    p.add_argument('--method', type=str, default='ODR', help=HELP_METHOD)
    p.add_argument('--hwp', action='store_true', help=HELP_HWP)
    p.add_argument('--show', action='store_true', help=HELP_SHOW)
    p.add_argument('-v', '--verbose', action='store_true', help=HELP_VERBOSE)


def add_analysis_subparser(subparsers):
    p = subparsers.add_parser("analysis", help=HELP_ANALYSYS)
    p.add_argument('name', type=str, help=HELP_SIM_NAME)
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
    add_plot_raw_subparser(subparsers)
    add_or_subparser(subparsers)
    add_analysis_subparser(subparsers)
    add_sim_subparser(subparsers)

    args = parser.parse_args(args=sys.argv[1:] or ['--help'])

    try:
        if args.command == 'phase_diff':
            setup_logger(args.verbose)
            phase_diff.plot_phase_difference_from_file(
                args.filepath,
                method=args.method,
                show=args.show
            )

        if args.command == 'OR':
            setup_logger(args.verbose)
            optical_rotation.optical_rotation(
                args.folder1,
                args.folder2,
                method=args.method,
                hwp=args.hwp
            )

        if args.command == 'plot_raw':
            setup_logger(args.verbose)
            raw.plot_raw(
                args.filepath,
                ch0=not args.no_ch0,
                ch1=not args.no_ch1,
                show=args.show
            )

        if args.command == 'analysis':
            setup_logger(args.verbose)
            analysis.main(args.name, show=args.show)

        if args.command == 'sim':
            setup_logger(args.verbose)
            simulator.main(
                args.name,
                method=args.method,
                reps=args.reps,
                step=args.step,
                samples=args.samples,
                show=args.show
            )

        if args.command == 'polarimeter':
            setup_logger(args.verbose)
            polarimeter.main(
                samples=args.samples,
                chunk_size=args.chunk_size,
                cycles=args.cycles,
                step=args.step,
                init_position=args.init_position,
                delay_position=args.delay_position,
                velocity=args.velocity,
                hwp_cycles=args.hwp_cycles,
                hwp_step=args.hwp_step,
                hwp_delay=args.hwp_delay,
                reps=args.reps,
                no_ch0=args.no_ch0,
                no_ch1=args.no_ch1,
                prefix=args.prefix,
                mock_esp=args.mock_esp,
                mock_adc=args.mock_adc,
                plot=args.plot,
                overwrite=args.overwrite
            )
    except ValueError as e:
        print("ERROR: {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
