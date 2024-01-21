import sys
import logging
import argparse

from rapp.commands import COMMANDS


logger = logging.getLogger(__name__)


APP_NAME = 'RAPP'
DESCRIPTION = 'Tools for measuring the rotation angle of the plane of polarization (RAPP).'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def setup_logger():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    # Hide logging from external libraries
    external_libs = ['matplotlib', 'PIL']
    for lib in external_libs:
        logging.getLogger(lib).setLevel(logging.ERROR)


class RAPPError(Exception):
    pass


class RAPP:
    def __init__(self, commands):
        self.parser = argparse.ArgumentParser(prog=APP_NAME, description=DESCRIPTION)
        self.subparsers = self.parser.add_subparsers(dest='command', help='available commands')

        self.commands = commands

        for command in self.commands.values():
            command.add_to_subparsers(self.subparsers)

    def parse_args(self, args):
        self.args = self.parser.parse_args(args=args or ['--help'])

        if self.args.command not in self.commands:
            raise RAPPError("Invalid RAPP command: {}".format(self.command))

        self.command = self.commands[self.args.command]

        if self.args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    def execute(self):
        return self.command.run(**self._get_command_args())

    def _get_command_args(self):
        command_args = vars(self.args)
        del command_args['command']
        del command_args['verbose']

        return command_args


def run(args):
    setup_logger()

    try:
        logger.info("Initializing RAPP")
        rapp = RAPP(COMMANDS)

        logger.info("Parsing RAPP arguments...")
        rapp.parse_args(args)

        logger.info("Running RAPP command: {}".format(rapp.args.command))
        rapp.execute()
    except RAPPError as e:
        rapp.logger.error(e)


def main():
    run(sys.argv[1:])


if __name__ == '__main__':
    main()
