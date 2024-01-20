from rapp import constants as ct

from rapp.analysis import analysis


HELP_ANALYSYS = "Tool for analyzing signals: noise, drift, etc."

HELP_NAME = (
    'name of the analysis. '
    'One of {}.'.format(analysis.ANALYSIS_NAMES)
)


def add_to_subparsers(subparsers):
    p = subparsers.add_parser("analysis", help=HELP_ANALYSYS)
    p.add_argument('name', type=str, help=HELP_NAME)
    p.add_argument('--show', action='store_true', help=ct.HELP_SHOW)
    p.add_argument('-v', '--verbose', action='store_true', help=ct.HELP_VERBOSE)


def run(**kwargs):
    analysis.main(**kwargs)
