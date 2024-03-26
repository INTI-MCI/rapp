import os
from rapp import constants as ct
from rapp.analysis import phase_diff


HELP_PHASE_DIFF = 'Tool for calculating phase difference from single polarimeter measurement.'

HELP_METHOD = 'phase difference calculation method (default: %(default)s).'
HELP_FILEPATH = 'file containing the measurements.'

EXAMPLE = "rapp phase_diff data/sine-range4V-632nm-cycles2-step1.0-samples50.txt"
EPILOG = "Example: {}".format(EXAMPLE)


def add_to_subparsers(subparsers):
    p = subparsers.add_parser("phase_diff", help=HELP_PHASE_DIFF, epilog=EPILOG)
    p.add_argument('filepath', type=str, help=HELP_FILEPATH)
    p.add_argument('--method', type=str, default='NLS', help=HELP_METHOD)
    p.add_argument('--show', action='store_true', help=ct.HELP_SHOW)
    p.add_argument('-v', '--verbose', action='store_true', help=ct.HELP_VERBOSE)


def run(filepath, **kwargs):
    if os.path.isdir(filepath):
        phase_diff.phase_difference_from_folder(filepath, **kwargs)
        return

    phase_diff.phase_difference_from_file(filepath, **kwargs)
