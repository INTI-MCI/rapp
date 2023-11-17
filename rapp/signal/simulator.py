import os
import logging
import numpy as np

from rapp import constants as ct
from rapp.utils import create_folder
from rapp.signal.plot import Plot
from rapp.signal.phase import phase_difference, PHASE_DIFFERENCE_METHODS


logger = logging.getLogger(__name__)


ANALYZER_VELOCITY = 4  # Degrees per second.
PHI = np.pi/4
AMPLITUDE_SIGNALS = 2   # Peak to peak amplitude
FACTOR_QUANTIZATION = 4
BITS_QUANTIZATION = 16  # Use signed values with this level of quantization
SAMPLES_PER_POSITION = 50

# Noise measured from dark current
# A0_NOISE = (-0.0004, 0.0003)
# A1_NOISE = (-0.003, 0.0001)

# Noise measured with laser ON
A0_NOISE = (1.9e-07, 0.00092)
A1_NOISE = (-1.7e-07, 0.00037)


np.random.seed(1)  # To make random simulations repeatable.


def harmonic(
    A: float = 2,
    n: int = 1,
    fc: int = 50,
    phi: float = 0,
    noise: tuple = None,
    all_positive: bool = False
) -> tuple:
    """Simulates a harmonic signal.

    Args:
        A: amplitude (peak) of the signal.
        n: number of cycles.
        fc: samples per cycle.
        phi: phase (radians).
        noise: (mu, sigma) of additive white gaussian noise.
        all_positive: if true, shifts the signal to the positive axis.

    Returns:
        The signal as an (xs, ys) tuple.
    """

    xs = np.linspace(0, 2 * np.pi * n, num=n * fc)

    signal = A * np.sin(xs + phi)

    additive_noise = np.zeros(xs.size)
    if noise is not None:
        mu, sigma = noise
        additive_noise = np.random.normal(loc=mu, scale=sigma, size=xs.size)

    signal = signal + additive_noise

    if all_positive:
        signal = signal + A

    return xs, signal


def quantize(values, max_value, bits, average_samples=1, signed=True):
    """Quantization of (simulated) values.

    Args:
        values: ndarray with float values
        max_value: maximum value of the scale
        bits: number of bits for quantization
        signed: allow using negative values

    Returns:
        Quantized values
    """
    max_q = 2 ** (bits - 1 if signed else bits)
    quantum = max_value / max_q
    quantum /= average_samples
    max_q *= average_samples
    q_values = np.round(values / quantum)
    q_values[q_values >= max_q - average_samples] = max_q - average_samples
    if signed:
        q_values[q_values < -max_q] = -max_q
    return q_values * quantum


def polarimeter_signal(cycles, fc, phi=0, a0_noise=None, a1_noise=None, **kwargs):
    """Simulates a pair of signals measured by the polarimeter."""
    if SAMPLES_PER_POSITION > 1:
        a0_noise = np.array(a0_noise)
        a0_noise[1] /= np.sqrt(SAMPLES_PER_POSITION)
        a1_noise = np.array(a1_noise)
        a1_noise[1] /= np.sqrt(SAMPLES_PER_POSITION)

    xs, s1 = harmonic(A=AMPLITUDE_SIGNALS / 2, n=cycles, fc=fc, noise=a0_noise, **kwargs)
    _, s2 = harmonic(A=AMPLITUDE_SIGNALS / 2, n=cycles, fc=fc, phi=-phi, noise=a1_noise, **kwargs)

    # Use quantized values
    s1 = quantize(
        s1,
        max_value=FACTOR_QUANTIZATION, bits=BITS_QUANTIZATION, average_samples=SAMPLES_PER_POSITION
    )
    s2 = quantize(
        s2,
        max_value=FACTOR_QUANTIZATION, bits=BITS_QUANTIZATION, average_samples=SAMPLES_PER_POSITION
    )

    # We divide angles by 2 because one cycle of the analyzer contains two cycles of the signal.
    return xs / 2, s1, s2


def samples_per_cycle(step=0.01):
    # Half cycle (180) of the analyzer is one full cycle of the signal.
    return int(180 / step)


def total_time(n_cycles):
    return n_cycles * (180 / ANALYZER_VELOCITY)


def plot_two_signals(phi, folder, s1_noise=None, s2_noise=None, show=False):
    print("")
    logger.info("TWO HARMONIC SIGNALS...")

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, folder=folder)

    xs, s1 = harmonic(noise=s1_noise)
    _, s2 = harmonic(phi=-phi, noise=s2_noise)

    label_template = "(φ1 - φ2)={}°.\n \t (µ, σ1)={} \n \t (µ, σ1)={}"
    label = label_template.format(round(phi, 2), s1_noise, s2_noise).expandtabs(11)

    plot.add_data(xs, s1, color='k', style='o-', label=label, xrad=True)
    plot.add_data(xs, s2, color='k', style='o-', xrad=True)
    plot.legend(loc='upper right', fontsize=10)

    plot.save(filename='sim_two_signals')

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def plot_error_vs_cycles(phi, folder, step=0.01, max_cycles=20, reps=1, show=False):
    print("")
    logger.info("PHASE DIFFERENCE VS # OF CYCLES")

    fc = samples_per_cycle(step=step)
    cycles_list = np.arange(1, max_cycles + 1, step=1)

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES,
        title="\nfc={}, step={} deg, reps={}".format(fc, step, reps), ysci=True,
        folder=folder
    )

    for method in PHASE_DIFFERENCE_METHODS:
        logger.info("Method: {}".format(method))

        errors = []
        for cycles in cycles_list:
            # Find RMSE using a number of repetitions of the simulation
            error = 0
            for rep in range(reps):
                xs, s1, s2 = polarimeter_signal(
                    cycles, fc, phi, A0_NOISE, A1_NOISE, all_positive=True
                )

                res = phase_difference(xs * 2, s1, s2, method=method)

                error += abs(phi - res.value) ** 2
            error /= reps
            error = np.sqrt(error)

            error_degrees = np.rad2deg(error)
            error_degrees_sci = "{:.2E}".format(error_degrees)

            errors.append(error_degrees)

            time = total_time(cycles) / 60
            logger.info(
                "cycles={}, fc={}, time={} m, φerr: {}, reps: {}."
                .format(cycles, fc, time, error_degrees_sci, reps))

        plot.add_data(cycles_list, errors, style='o-', label=method)

    plot.legend()

    plot.save(filename="sim_phi_error_vs_cycles")

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def plot_error_vs_step(phi, folder, cycles=20, show=False):
    print("")
    logger.info("PHASE DIFFERENCE ERROR VS STEP")
    time_min = total_time(cycles) / 60

    steps = np.arange(0.1, 1.1, step=0.1)[::-1]

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_STEP,
        title="cycles={}, time={} min.".format(cycles, time_min), ysci=True, folder=folder)

    for method in PHASE_DIFFERENCE_METHODS:
        logger.info("Method: {}".format(method))

        errors = []
        for step in steps:
            fc = samples_per_cycle(step=step)
            xs, s1, s2 = polarimeter_signal(cycles, fc, phi, A0_NOISE, A1_NOISE)
            res = phase_difference(xs * 2, s1, s2, method=method)

            error_degrees = np.rad2deg(abs(phi - res.value))
            error_degrees_sci = "{:.2E}".format(error_degrees)

            errors.append(error_degrees)
            step = round(step, 1)
            logger.info(
                "cycles={}, fc={}, step={}, φerr: {}."
                .format(cycles, fc, step, error_degrees_sci))

        plot.add_data(steps, errors, style='o-', label=method)

    plot.legend()

    plot.save(filename="sim_phi_error_vs_step")

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def plot_phase_diff(phi, folder, cycles=10, step=0.01, show=False):
    print("")
    logger.info("PHASE DIFFERENCE OF TWO SIMULATED SIGNALS")

    fc = samples_per_cycle(step=step)

    logger.info("Simulating signals...")
    xs, s1, s2 = polarimeter_signal(cycles, fc, phi, A0_NOISE, A1_NOISE, all_positive=True)

    logger.info("Calculating phase difference...")
    res = phase_difference(xs * 2, s1, s2, method='fit')

    error = abs(phi - res.value)
    error_degrees = np.rad2deg(error)

    logger.info("Detected phase difference: {}".format(np.rad2deg(res.value)))
    logger.info("cycles={}, fc={}, step={}, φerr: {}.".format(cycles, fc, step, error_degrees))

    label = (
        "fc={}. \n".format(fc),
        "step={} deg. \n".format(step),
        "# cycles={}. \n".format(cycles),
        "|φ1 - φ2| = {}°. \n".format(round(np.rad2deg(phi)))
    )

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, folder=folder)

    markevery = int(fc / 180) if fc >= 180 else 1  # for visualization purposes, show less points.

    plot.add_data(
        xs, s1,
        ms=6, color='k', mew=0.5, xrad=True, markevery=markevery, alpha=0.8, label=label
    )

    plot.add_data(
        xs, s2,
        ms=6, color='k', mew=0.5, xrad=True, markevery=markevery, alpha=0.8
    )

    label = "φerr = {}.".format(round(error_degrees, 5))
    plot.add_data(res.fitx / 2, res.fits1, style='-', color='k', lw=1.5, xrad=True)
    plot.add_data(res.fitx / 2, res.fits2, style='-', color='k', lw=1.5, xrad=True, label=label)

    plot._ax.set_xlim(0, 1)

    plot.legend(loc='upper right')

    plot.save(filename="sim_phase_diff_fit")

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def main(sim, reps=1, show=False):
    print("")
    logger.info("STARTING SIMULATIONS...")

    logger.info("PHASE DIFFERENCE: {} degrees.".format(np.rad2deg(PHI)))
    logger.info("ANALYZER VELOCITY: {} degrees per second.".format(ANALYZER_VELOCITY))

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    # TODO: add another subparser and split these options in different commands with parameters
    if sim not in ['all', 'two_signals', 'error_vs_cycles', 'error_vs_step', 'phase_diff']:
        raise ValueError("Simulation with name {} not implemented".format(sim))

    if sim in ['all', 'two_signals']:
        plot_two_signals(PHI, output_folder, s1_noise=A0_NOISE, s2_noise=A1_NOISE, show=show)

    if sim in ['all', 'error_vs_cycles']:
        plot_error_vs_cycles(PHI, output_folder, max_cycles=20, step=0.01, reps=reps, show=show)

    if sim in ['all', 'error_vs_step']:
        plot_error_vs_step(PHI, output_folder, cycles=20, show=show)

    if sim in ['all', 'phase_diff']:
        plot_phase_diff(PHI, output_folder, cycles=10, step=0.01, show=show)


if __name__ == '__main__':
    main(sim='error_vs_cycles', show=True)
