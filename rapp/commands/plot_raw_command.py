from rapp import constants as ct

from rapp.analysis import raw

HELP_PLOT_ROW = "Tool for plotting raw measurements."

HELP_FILEPATH = 'file containing the measurements.'
HELP_NOCH0 = 'excludes channel 0 from measurement (default: %(default)s).'
HELP_NOCH1 = 'excludes channel 1 from measurement (default: %(default)s).'

EXAMPLE = "rapp plot_raw data/sine-range4V-632nm-cycles2-step1.0-samples50.txt"
EPILOG = "Example: {}".format(EXAMPLE)


def add_to_subparsers(subparsers):
    p = subparsers.add_parser("plot_raw", help=HELP_PLOT_ROW, epilog=EPILOG)
    p.add_argument('filepath', type=str, help=HELP_FILEPATH)
    p.add_argument('--no-ch0', action='store_true', help=HELP_NOCH0)
    p.add_argument('--no-ch1', action='store_true', help=HELP_NOCH1)
    p.add_argument('--show', action='store_true', help=ct.HELP_SHOW)
    p.add_argument('-v', '--verbose', action='store_true', help=ct.HELP_VERBOSE)


def run(**kwargs):
    raw.plot_raw_from_file(**kwargs)
