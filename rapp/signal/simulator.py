import os
import logging
import numpy as np

from rapp import constants as ct
from rapp.utils import create_folder
from rapp.signal.plot import Plot
from rapp.signal.phase import phase_difference


logger = logging.getLogger(__name__)


PHI = np.pi / 4         # Phase difference.
ANALYZER_VELOCITY = 4   # Degrees per second.

ADC_MAXV = 4.096
ADC_BITS = 16     # Use signed values with this level of quantization.

# Noise measured from dark current
# A0_NOISE = (-0.0004, 0.0003)
# A1_NOISE = (-0.003, 0.0001)

# Noise measured with laser ON
A0_NOISE = [1.9e-07, 0.00092]
A1_NOISE = [-1.7e-07, 0.00037]


np.random.seed(1)  # To make random simulations repeatable.


def harmonic(
    A: float = 2,
    cycles: int = 1,
    fc: int = 50,
    phi: float = 0,
    noise: tuple = None,
    all_positive: bool = False
) -> tuple:
    """Simulates a harmonic signal.

    Args:
        A: amplitude (peak) of the signal.
        cycles: number of cycles.
        fc: samples per cycle.
        phi: phase (radians).
        noise: (mu, sigma) of additive white gaussian noise.
        all_positive: if true, shifts the signal to the positive axis.

    Returns:
        The signal as an (xs, ys) tuple.
    """

    xs = np.linspace(0, 2 * np.pi * cycles, num=cycles * fc)

    signal = A * np.sin(xs + phi)

    additive_noise = np.zeros(xs.size)
    if noise is not None:
        mu, sigma = noise
        additive_noise = np.random.normal(loc=mu, scale=sigma, size=xs.size)

    signal = signal + additive_noise

    if all_positive:
        signal = signal + A

    return xs, signal


def quantize(signal, max_v=ADC_MAXV, bits=ADC_BITS, samples=1, signed=True):
    """Performs quantization of a (simulated) signal.

    Args:
        signal (np.array): values to quantize (in Volts).
        max_v: maximum value of ADC scale [0, max_v] (in Volts).
        bits: number of bits for quantization.
        signed: allow using negative values.

    Returns:
        Quantized signal.
    """
    max_q = 2 ** (bits - 1 if signed else bits)
    q_factor = max_v / max_q

    q_factor /= samples
    max_q *= samples

    q_signal = np.round(signal / q_factor)
    q_signal[q_signal >= max_q] = max_q

    if signed:
        q_signal[q_signal < -max_q] = -max_q

    logger.debug("Quantization: bits={}, factor={}".format(bits, q_factor))

    return q_signal * q_factor


def polarimeter_signal(
    phi=0, samples=50, a0_noise=None, a1_noise=None, bits=ADC_BITS, max_v=ADC_MAXV, **kwargs
):
    """Simulates a pair of signals measured by the polarimeter.

    Args:
        phi: phase difference between signals.
        samples: number of samples in each angle.
        a0_noise: (mu, sigma) of additive white gaussian noise of channel 0.
        a1_noise: (mu, sigma) of additive white gaussian noise of channel 1.
        bits: number of bits of the signal.
        **kwargs: any other keyword argument to be passed 'harmonic' function.

    Returns:
        the X-coordinates and S1, S2 Y-coordinates as (X, S1, S2) 3-tuple.
    """
    a0_noise[1] /= np.sqrt(samples)
    a1_noise[1] /= np.sqrt(samples)

    xs, s1 = harmonic(noise=a0_noise, **kwargs)
    _, s2 = harmonic(phi=-phi, noise=a1_noise, **kwargs)

    # Use quantized values
    s1 = quantize(s1, max_v=max_v, bits=bits, samples=samples)
    s2 = quantize(s2, max_v=max_v, bits=bits, samples=samples)

    # We divide angles by 2 because one cycle of the analyzer contains two cycles of the signal.
    return xs / 2, s1, s2


def samples_per_cycle(step=0.01):
    # Half cycle (180) of the analyzer is one full cycle of the signal.
    return int(180 / step)


def total_time(n_cycles):
    return n_cycles * (180 / ANALYZER_VELOCITY)


def n_simulations(n=1, method='fit', **kwargs):
    """Performs n simulations and returned a list of n errors.

    Args:
        n: number of simulations.
        method: phase difference alculation method.
        *kwargs: arguments for polarimeter_signal function.
    """
    errors = []
    for i in range(n):
        xs, s1, s2 = polarimeter_signal(**kwargs)
        res = phase_difference(xs * 2, s1, s2, method=method)
        errors.append(res.value)

    return errors


def plot_two_signals(phi, folder, s1_noise=None, s2_noise=None, show=False):
    print("")
    logger.info("TWO HARMONIC SIGNALS...")

    label_template = "(φ1 - φ2)={}°.\n(µ, σ1)={}\n(µ, σ1)={}"
    label = label_template.format(round(phi, 2), s1_noise, s2_noise).expandtabs(11)

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, folder=folder)

    plot.set_title(label)
    xs, s1, s2 = polarimeter_signal(
        A=2, cycles=1, phi=phi, fc=360, samples=1, a0_noise=s1_noise, a1_noise=s2_noise,
        bits=5, max_v=ADC_MAXV, all_positive=True
    )

    plot.add_data(xs, s1, style='-', color='k', xrad=True)
    plot.add_data(xs, s2, style='-', xrad=True)
    plot._ax.set_ylim(0, ADC_MAXV)

    plot.save(filename='sim_two_signals')

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def plot_error_vs_cycles(phi, folder, samples=50, max_cycles=10, reps=1, show=False):
    print("")
    logger.info("PHASE DIFFERENCE VS # OF CYCLES")

    cycles_list = np.arange(1, max_cycles + 1, step=1)

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES,
        title="reps={}".format(reps), ysci=True, xint=True,
        folder=folder
    )

    for step in [0.01, 0.5, 1]:
        fc = samples_per_cycle(step=step)
        logger.info("Method: {}, fc={}, reps={}".format('fit', fc, reps))

        errors = []
        for cycles in cycles_list:
            n_errors = n_simulations(
                n=reps, method='fit', cycles=cycles, fc=fc, phi=phi,
                samples=samples, a0_noise=A0_NOISE, a1_noise=A1_NOISE, all_positive=True
            )

            # RMSE
            error_rad = np.sqrt(sum([abs(phi - e) ** 2 for e in n_errors]) / reps)
            error_degrees = np.rad2deg(error_rad)
            error_degrees_sci = "{:.2E}".format(error_degrees)

            errors.append(error_degrees)

            time = total_time(cycles) / 60
            logger.info("cycles={}, time={} m, φerr: {}.".format(cycles, time, error_degrees_sci))

        label = "step={}, fc={}".format(step, fc)
        plot.add_data(cycles_list, errors, style='-', lw=2, label=label)

    plot.legend()

    plot.save(filename="sim_phi_error_vs_cycles")

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def plot_error_vs_step(phi, folder, samples=1, cycles=5, reps=1, show=False):
    print("")
    logger.info("PHASE DIFFERENCE ERROR VS STEP")
    time_min = total_time(cycles) / 60

    steps = np.arange(0.1, 1.1, step=0.1)[::-1]

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_STEP,
        title="cycles={}, time={} min.".format(cycles, time_min), ysci=True, folder=folder)

    method = 'fit'

    logger.info("Method: {}".format(method))

    errors = []
    for step in steps:
        fc = samples_per_cycle(step=step)
        n_errors = n_simulations(
            n=reps, method='fit', cycles=cycles, fc=fc, phi=phi,
            samples=samples, a0_noise=A0_NOISE, a1_noise=A1_NOISE, all_positive=True
        )

        # RMSE
        error_rad = np.sqrt(sum([abs(phi - e) ** 2 for e in n_errors]) / reps)
        error_degrees = np.rad2deg(error_rad)
        error_degrees_sci = "{:.2E}".format(error_degrees)

        errors.append(error_degrees)
        logger.info("fc={}, step={}, φerr: {}.".format(fc, round(step, 1), error_degrees_sci))

    plot.add_data(steps, errors, style='o-', color='k', label=method)

    plot.legend()

    plot.save(filename="sim_phi_error_vs_step")

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def plot_phase_diff(phi, folder, samples=50, cycles=10, step=0.01, show=False):
    print("")
    logger.info("PHASE DIFFERENCE OF TWO SIMULATED SIGNALS")

    fc = samples_per_cycle(step=step)

    logger.info("Simulating signals...")
    xs, s1, s2 = polarimeter_signal(
        cycles=cycles, fc=fc, phi=phi, samples=samples,
        a0_noise=A0_NOISE, a1_noise=A1_NOISE, all_positive=True
    )

    logger.info("Calculating phase difference...")
    res = phase_difference(xs * 2, s1, s2, method='fit')

    error = abs(phi - res.value)
    error_degrees = np.rad2deg(error)

    logger.info("Detected phase difference: {}".format(np.rad2deg(res.value)))
    logger.info("cycles={}, fc={}, step={}, φerr: {}.".format(cycles, fc, step, error_degrees))

    label = (
        "fc={}. \n"
        "step={} deg. \n"
        "# cycles={}. \n"
        "|φ1 - φ2| = {}°. \n"
    ).format(fc, step, cycles, round(np.rad2deg(phi)))

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


def main(sim, reps=1, samples=1, show=False):
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
        plot_two_signals(np.pi / 2, output_folder, s1_noise=A0_NOISE, s2_noise=A1_NOISE, show=show)

    if sim in ['all', 'error_vs_cycles']:
        plot_error_vs_cycles(PHI, output_folder, samples, max_cycles=10, reps=reps, show=show)

    if sim in ['all', 'error_vs_step']:
        plot_error_vs_step(PHI, output_folder, samples, cycles=10, show=show)

    if sim in ['all', 'phase_diff']:
        plot_phase_diff(PHI, output_folder, samples, cycles=10, step=0.01, show=show)


if __name__ == '__main__':
    main(sim='error_vs_cycles', show=True)
