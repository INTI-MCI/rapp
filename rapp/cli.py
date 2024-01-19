import sys
import logging
import argparse

from rapp.log import setup_logger

from rapp.commands import (
    polarimeter_command,
    phase_diff_command,
    analysis_command,
    plot_raw_command,
    optical_rotation_command,
    simulations_command,
)


APP_NAME = 'RAPP'
DESCRIPTION = 'Tools for measuring the rotation angle of the plane of polarization (RAPP).'


COMMANDS = {
    'analysis': analysis_command,
    'OR': optical_rotation_command,
    'phase_diff': phase_diff_command,
    'plot_raw': plot_raw_command,
    'polarimeter': polarimeter_command,
    'sim': simulations_command,
}


class RAPPError(Exception):
    pass


def add_commands(subparsers):
    for command in COMMANDS.values():
        command.add_to_subparsers(subparsers)


def get_command(command):
    if command not in COMMANDS:
        raise RAPPError("Invalid RAPP command: {}".format(command))

    return COMMANDS[command]


def get_command_args(args):
    command_args = vars(args)
    del command_args['command']
    del command_args['verbose']

    return command_args


def run(args):
    parser = argparse.ArgumentParser(prog=APP_NAME, description=DESCRIPTION)
    subparsers = parser.add_subparsers(dest='command', help='available commands')

    add_commands(subparsers)

    args = parser.parse_args(args=args or ['--help'])

    setup_logger(args.verbose)
    command = get_command(args.command)
    command_args = get_command_args(args)

    logger = logging.getLogger(__name__)

    try:
        command.run(**command_args)
    except RAPPError as e:
        logger.error(e)


def main():
    run(sys.argv[1:])


if __name__ == '__main__':
    main()
