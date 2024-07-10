import os
import argparse
from pathlib import Path

from rapp import constants as ct
from rapp.utils import create_folder, json_load
from rapp.simulations import simulation_factory


HELP = "Tool for making numerical simulations."
EXAMPLE = "rapp sim --name error_vs_samples --show"
EPILOG = "Example: {}".format(EXAMPLE)

HELP_CONFIG_FILE = "JSON config file to use (optional)."
HELP_NAME = "name of the simulation to run."

HELP_REPS = "number of repetitions in each simulated iteration."
HELP_ANGLE = "Angle between the two planes of polarization."
HELP_METHOD = "phase difference calculation method."
HELP_CYCLES = "n° of cycles (or maximum n° of cycles) to run."
HELP_STEP = "motion step of the rotating analyzer."
HELP_SAMPLES = "n° of samples per angle."
HELP_DYNAMIC_RANGE = "percentage of dynamic range to simulate (between [0, 1])."
HELP_MAX_V = "maximum voltage of dynamic range."
HELP_DISTORTION = "amount of distortion to add to CH0."


def add_to_subparsers(subparsers):
    p = subparsers.add_parser("sim", help=HELP, epilog=EPILOG, argument_default=argparse.SUPPRESS)
    p.add_argument("-c", "--config-file", type=Path, metavar="", help=HELP_CONFIG_FILE)
    p.add_argument("--name", metavar="", help=HELP_NAME)

    p = p.add_argument_group("simulation specific parameters")
    p.add_argument("--reps", type=int, metavar="", help=HELP_REPS)
    p.add_argument("--angle", type=float, metavar="", help=HELP_METHOD)
    p.add_argument("--method", type=str, metavar="", help=HELP_METHOD)
    p.add_argument("--cycles", type=int, metavar="", help=HELP_CYCLES)
    p.add_argument("--step", type=float, metavar="", help=HELP_STEP)
    p.add_argument("--samples", type=int, metavar="", help=HELP_SAMPLES)
    p.add_argument("--dynamic-range", type=float, metavar="", help=HELP_DYNAMIC_RANGE)
    p.add_argument("--max-v", type=float, metavar="", help=HELP_MAX_V)
    p.add_argument("--k", type=float, metavar="", help=HELP_DISTORTION)
    p.add_argument("--show", action="store_true", help=ct.HELP_SHOW)


def run(name=None, config_file=None, work_dir=ct.WORK_DIR, **kwargs):
    output_folder = os.path.join(work_dir, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    if config_file is None and name is None:
        raise ValueError("You need to provide a config file or a name.")

    if config_file is not None:
        config = json_load(config_file)
    else:
        config = {}

    if name is not None:
        # Ensure name is in config.
        config[name] = config.get(name, {})

        # Select only one simulation.
        config = {k: v for k, v in config.items() if k == name}

    # Overwrite config with CLI params.
    for k, v in kwargs.items():
        for sim, sim_config in config.items():
            sim_config[k] = v

    # Get simulations objects.
    sims = {simulation_factory.create(name): config for name, config in config.items()}

    # Execute simulations.
    for sim, config in sims.items():
        sim.run(output_folder, **config)


if __name__ == "__main__":
    run(name="error_vs_step", show=True)
