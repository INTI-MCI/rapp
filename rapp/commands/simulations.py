import os

from rapp import constants as ct
from rapp.utils import create_folder
from rapp.simulations import simulations_builder


HELP = "Tool for making numerical simulations."
EXAMPLE = "rapp sim error_vs_samples --show"
EPILOG = "Example: {}".format(EXAMPLE)

HELP_ANGLE = 'Angle between the two planes of polarization'
HELP_CYCLES = 'n° of cycles (or maximum n° of cycles) to run (default: %(default)s).'
HELP_NAME = 'names of the simulations to run.'
HELP_METHOD = 'phase difference calculation method (default: %(default)s).'
HELP_REPS = 'number of repetitions in each simulated iteration (default: %(default)s).'
HELP_SAMPLES = 'n° of samples per angle.'
HELP_STEP = 'motion step of the rotating analyzer (default: %(default)s).'


def add_to_subparsers(subparsers):
    p = subparsers.add_parser("sim", help=HELP, epilog=EPILOG)
    p.add_argument('--names', nargs='+', metavar='', help=HELP_NAME)
    p.add_argument('--method', type=str, default='NLS', metavar='', help=HELP_METHOD)
    p.add_argument('--angle', type=float, default=22.5, metavar='', help=HELP_METHOD)
    p.add_argument('--samples', type=int, default=50, metavar='', help=HELP_SAMPLES)
    p.add_argument('--cycles', type=int, default=1, metavar='', help=HELP_CYCLES)
    p.add_argument('--step', type=float, default=1, metavar='', help=HELP_STEP)
    p.add_argument('--reps', type=int, default=1, metavar='', help=HELP_REPS)
    p.add_argument('--show', action='store_true', help=ct.HELP_SHOW)
    p.add_argument('-v', '--verbose', action='store_true', help=ct.HELP_VERBOSE)


def run(names, work_dir=ct.WORK_DIR, **kwargs):
    output_folder = os.path.join(work_dir, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    simulations = simulations_builder.build(names)

    for simulation in simulations:
        simulation.run(output_folder, **kwargs)


if __name__ == '__main__':
    run(name='error_vs_step', show=True)
