from rapp.simulations import (
    error_vs_cycles,
    error_vs_range,
    error_vs_resolution,
    error_vs_samples,
    error_vs_step,
    error_vs_phase,
    error_vs_all,
    one_phase_diff,
    pvalue_vs_range,
    simulation_steps,
)

SIMULATIONS = {
    "error_vs_cycles": error_vs_cycles,
    "error_vs_range": error_vs_range,
    "error_vs_resolution": error_vs_resolution,
    "error_vs_samples": error_vs_samples,
    "error_vs_step": error_vs_step,
    "error_vs_phase": error_vs_phase,
    "error_vs_all": error_vs_all,
    "one_phase_diff": one_phase_diff,
    "pvalue_vs_range": pvalue_vs_range,
    "sim_steps": simulation_steps,
}


def create(name):
    if name not in SIMULATIONS:
        raise ValueError("Simulation with name {} not implemented".format(name))

    return SIMULATIONS[name]
