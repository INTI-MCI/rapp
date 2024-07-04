import argparse

from rapp import constants as ct
from rapp import polarimeter


NAME = "polarimeter"
HELP = "Tool for measuring signals with the polarimeter."
EXAMPLE = "rapp polarimeter --cycles 1 --step 30 --samples 10 --delay-position 0"
EPILOG = "Example: {}".format(EXAMPLE)

HELP_SAMPLES = 'n° of samples per angle, default (%(default)s) gives 10 cycles of 50 Hz noise.'
HELP_CYCLES = 'n° of cycles to run (default: %(default)s).'
HELP_STEP = 'step of the rotating analyzer (default: %(default)s). If cycles==0, step is ignored.'
HELP_DELAY_POSITION = 'delay (in seconds) after changing analyzer position (default: %(default)s).'
HELP_VELOCITY = 'velocity of the analyzer in deg/s (default: %(default)s).'
HELP_ACCELERATION = 'acceleration of the analyzer in deg/s^2 (default: %(default)s).'
HELP_DECELERATION = 'deceleration of the analyzer in deg/s^2 (default: %(default)s).'
HELP_INIT_P = 'initial position of the analyzer in deg (default: %(default)s).'
HELP_NOCH0 = 'excludes channel 0 from measurement (default: %(default)s).'
HELP_NOCH1 = 'excludes channel 1 from measurement (default: %(default)s).'
HELP_CHUNK_SIZE = 'measure data in chunks of this size. If 0, no chunks (default: %(default)s).'
HELP_PREFIX = 'prefix for the folder in which to write results (default: %(default)s).'
HELP_TEST_ESP = 'use ESP mock object. (default: %(default)s).'
HELP_TEST_ADC = 'use ADC mock object. (default: %(default)s).'
HELP_TEST_PM100 = 'use PM100 mock object. (default: %(default)s).'
HELP_DISABLE_PM100 = 'disables the PM100 normalization detector. (default: %(default)s).'
HELP_OVERWRITE = 'whether to overwrite existing files without asking (default: %(default)s).'
HELP_REPS = 'Number of repetitions (default: %(default)s).'
HELP_WORKDIR = 'folder to use as working directory (default: %(default)s)'
HELP_DELAY_POSITION = 'delay (in seconds) after changing analyzer position (default: %(default)s).'
HELP_MC_WAIT = 'time to wait (in seconds) before re-connecting the MC (default: %(default)s).'


HELP_HWP_CYCLES = 'n° of cycles of the HW plate (default: %(default)s).'
HELP_HWP_STEP = 'motion step of the rotating HW plate (default: %(default)s).'
HELP_HWP_DELAY = 'delay (in seconds) after changing HW plate position (default: %(default)s).'
HELP_HWP = 'whether the measurement was done with a half wave plate (default: %(default)s).'


def formatter(prog):
    return argparse.RawTextHelpFormatter(prog, max_help_position=50)


def add_to_subparsers(subparsers):
    p = subparsers.add_parser(NAME, help=HELP, epilog=EPILOG, formatter_class=formatter)
    p.add_argument('--samples', type=int, default=169, help=HELP_SAMPLES)
    p.add_argument('--chunk-size', type=int, default=500, metavar='', help=HELP_CHUNK_SIZE)
    p.add_argument('--reps', type=int, default=1, metavar='', help=HELP_REPS)
    p.add_argument('--mc-wait', type=float, default=15, metavar='', help=HELP_MC_WAIT)
    p.add_argument('--prefix', type=str, default='test', metavar='', help=HELP_PREFIX)
    p.add_argument('--mock-esp', action='store_true', help=HELP_TEST_ESP)
    p.add_argument('--mock-pm100', action='store_true', help=HELP_TEST_PM100)
    p.add_argument('--disable-pm100', action='store_true', help=HELP_DISABLE_PM100)
    p.add_argument('--work-dir', type=str, metavar='', default=ct.WORK_DIR, help=HELP_WORKDIR)
    p.add_argument('-ow', '--overwrite', action='store_true', help=HELP_OVERWRITE)

    g = p.add_argument_group('ADC')
    g.add_argument('--mock-adc', action='store_true', help=HELP_TEST_ADC)
    g.add_argument('--no-ch0', action='store_true', help=HELP_NOCH0)
    g.add_argument('--no-ch1', action='store_true', help=HELP_NOCH1)

    g = p.add_argument_group('Analyzer')
    g.add_argument('--cycles', type=float, default=0, help=HELP_CYCLES)
    g.add_argument('--step', type=float, default=45, help=HELP_STEP)
    g.add_argument('--delay-position', type=float, default=0, metavar='', help=HELP_DELAY_POSITION)
    g.add_argument('--velocity', type=float, default=2.5, metavar='', help=HELP_VELOCITY)
    g.add_argument('--acceleration', type=float, default=6.5, metavar='', help=HELP_ACCELERATION)
    g.add_argument('--deceleration', type=float, default=4.5, metavar='', help=HELP_DECELERATION)

    g = p.add_argument_group('Half Wate Plate')
    g.add_argument('--hwp-cycles', type=float, default=0, metavar='', help=HELP_HWP_CYCLES)
    g.add_argument('--hwp-step', type=float, default=45, metavar='', help=HELP_HWP_STEP)
    g.add_argument('--hwp-delay-position', type=float, default=5, metavar='', help=HELP_HWP_DELAY)


def run(**config):
    polarimeter.run(**config)
