from rapp import constants as ct
from rapp.analysis import optical_rotation

HELP_OR = 'Tool for calculating optical rotation from initial phase and final phase measurements.'

HELP_METHOD = 'phase difference calculation method (default: %(default)s).'
HELP_FOLDER_WITHOUT_SAMPLE = 'folder containing the measurements without sample.'
HELP_FOLDER_WITH_SAMPLE = 'folder containing the measurements with sample.'
HELP_HWP = 'whether the measurement was done with a half wave plate (default: %(default)s).'


EXAMPLE = "rapp or data/28-12-2023/hwp0 data/28-12-2023/hwp29/"
EPILOG = "Example: {}".format(EXAMPLE)


def add_to_subparsers(subparsers):
    p = subparsers.add_parser("OR", help=HELP_OR, epilog=EPILOG)
    p.add_argument('folder1', type=str, help=HELP_FOLDER_WITHOUT_SAMPLE)
    p.add_argument('folder2', type=str, help=HELP_FOLDER_WITH_SAMPLE)
    p.add_argument('--method', type=str, default='ODR', help=HELP_METHOD)
    p.add_argument('--hwp', action='store_true', help=HELP_HWP)
    p.add_argument('-v', '--verbose', action='store_true', help=ct.HELP_VERBOSE)


def run(**kwargs):
    optical_rotation.optical_rotation(**kwargs)
