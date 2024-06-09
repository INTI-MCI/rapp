import sys
import os
import logging
import argparse

from pathlib import Path

from rich.logging import RichHandler

from rapp.commands import COMMANDS


logger = logging.getLogger(__name__)


class CLI:
    """Generic command-line interface."""
    HELP_VERBOSE = 'whether to run with DEBUG log level (default: %(default)s).'

    def __init__(self, args):
        self._args = args

        self._add_parser()
        self._add_subparsers()
        self._parse_args()
        self._setup_logger()

    def execute(self):
        config = vars(self.args)

        return self.command.run(**config)

    def _add_parser(self):
        self.parser = argparse.ArgumentParser(
            prog=self.NAME,
            description=self.DESCRIPTION,
            formatter_class=self.formatter())

        self.parser.add_argument('-v', '--verbose', action='store_true', help=self.HELP_VERBOSE)

    def _add_subparsers(self):

        self.subparsers = self.parser.add_subparsers(dest='command', help='available commands')

        for command in self.COMMANDS.values():
            command.add_to_subparsers(self.subparsers)

    def _parse_args(self):
        self.args, self.extra_args = self.parser.parse_known_args(args=self._args or ['--help'])

        if self.args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        self.command = self.COMMANDS[self.args.command]

        del self.args.verbose
        del self.args.command

    def _setup_logger(self):
        rh = RichHandler(level="NOTSET")
        rh.setFormatter(logging.Formatter(self.LOG_FORMAT_RICH))

        os.makedirs(self.args.work_dir, exist_ok=True)
        log_file = Path(self.args.work_dir).joinpath("{}.log".format(self.NAME))

        logging.basicConfig(
            level=logging.INFO,
            format=self.LOG_FORMAT, handlers=[logging.FileHandler(log_file), rh])

        for module in self.SUPRESS_LOG:
            logging.getLogger(module).setLevel(logging.ERROR)


class RAPPError(Exception):
    pass


class RAPP(CLI):
    NAME = 'RAPP'
    DESCRIPTION = 'Tools for measuring the rotation angle of the plane of polarization (RAPP).'
    LOG_FORMAT_RICH = '%(name)s - %(message)s'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # packages / moudules for which to supress any log level except ERROR.
    SUPRESS_LOG = ['matplotlib', 'PIL']

    COMMANDS = COMMANDS

    @staticmethod
    def formatter():
        def formatter(prog):
            return argparse.RawTextHelpFormatter(prog, max_help_position=50)

        return formatter


def run(args):
    RAPP(args).execute()


def main():
    run(sys.argv[1:])


if __name__ == '__main__':
    main()
