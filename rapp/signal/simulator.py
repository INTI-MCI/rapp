import os
import logging

import numpy as np
import pandas as pd


from rapp import constants as ct
from rapp.utils import create_folder
from rapp.signal.plot import Plot
from rapp.signal.phase import phase_difference, PHASE_DIFFERENCE_METHODS


logger = logging.getLogger(__name__)


PHI = np.pi / 4         # Phase difference.
ANALYZER_VELOCITY = 4   # Degrees per second.

ADC_MAXV = 4.096
ADC_BITS = 16     # Use signed values with this level of quantization.

ARDUINO_MAXV = 5
ARDUINO_BITS = 10

# Noise measured from dark current
# A0_NOISE = (-0.0004, 0.0003)
# A1_NOISE = (-0.003, 0.0001)

# Noise measured with laser ON
A0_NOISE = [1.9e-07, 0.00092]
A1_NOISE = [-1.7e-07, 0.00037]

SIMULATIONS = [
    'all', 'two_signals', 'methods',
    'error_vs_cycles', 'error_vs_res', 'phase_diff', 'error_vs_range'
]


np.random.seed(1)  # To make random simulations repeatable.


def harmonic(
    A: float = 2,
    cycles: int = 1,
    fc: int = 50,
    fa: int = 1,
    phi: float = 0,
    noise: tuple = None,
    all_positive: bool = False
) -> tuple:
    """Simulates a harmonic signal.

    Args:
        A: amplitude (peak) of the signal.
        cycles: number of cycles.
        fc: samples per cycle.
        fa: samples per angle.
        phi: phase (radians).
        noise: (mu, sigma) of additive white gaussian noise.
        all_positive: if true, shifts the signal to the positive axis.

    Returns:
        The signal as an (xs, ys) tuple.
    """

    xs = np.linspace(0, 2 * np.pi * cycles, num=int(cycles * fc))
    xs = np.repeat(xs, fa)

    signal = A * np.sin(xs + phi)

    additive_noise = np.zeros(xs.size)
    if noise is not None:
        mu, sigma = noise
        additive_noise = np.random.normal(loc=mu, scale=sigma, size=xs.size)

    signal = signal + additive_noise

    if all_positive:
        signal = signal + A

    return xs, signal


def quantize(
    signal: np.array, max_v: float = ADC_MAXV, bits: int = ADC_BITS, signed=True
) -> np.array:
    """Performs quantization of a (simulated) signal.

    Args:
        signal: values to quantize (in Volts).
        max_v: maximum value of ADC scale [0, max_v] (in Volts).
        bits: number of bits for quantization.
        signed: if true, allows using negative values.

    Returns:
        Quantized signal.
    """
    max_q = 2 ** (bits - 1 if signed else bits)
    q_factor = max_v / max_q

    q_signal = np.round(signal / q_factor)

    q_signal[q_signal > max_q - 1] = max_q - 1
    q_signal[q_signal < -max_q] = -max_q

    logger.debug("Quantization: bits={}, factor={}".format(bits, q_factor))

    return q_signal * q_factor


def polarimeter_signal(
    phi=0, a0_noise=None, a1_noise=None, bits=ADC_BITS, max_v=ADC_MAXV, **kwargs
):
    """Simulates a pair of signals measured by the polarimeter.

    Args:
        phi: phase difference between signals (radians).
        a0_noise: (mu, sigma) of additive white gaussian noise of channel 0.
        a1_noise: (mu, sigma) of additive white gaussian noise of channel 1.
        bits: number of bits of the signal.
        max_v: float number corresponding to the maximum value of ADC scale [0, max_v] (in Volts).
        **kwargs: any other keyword argument to be passed 'harmonic' function.

    Returns:
        the X-coordinates and S1, S2 Y-coordinates as (X, S1, S2) 3-tuple.
    """
    xs, s1 = harmonic(noise=a0_noise, **kwargs)
    _, s2 = harmonic(phi=-phi, noise=a1_noise, **kwargs)

    # Use quantized values
    s1 = quantize(s1, max_v=max_v, bits=bits)
    s2 = quantize(s2, max_v=max_v, bits=bits)

    # We divide angles by 2 because one cycle of the analyzer contains two cycles of the signal.
    return xs / 2, s1, s2


def samples_per_cycle(step=0.01):
    # Half cycle (180) of the analyzer is one full cycle of the signal.
    return int(180 / step)


def total_time(n_cycles):
    return n_cycles * (180 / ANALYZER_VELOCITY)


def n_simulations(n=1, method='curve_fit', **kwargs):
    """Performs n simulations and returned a list of n errors.

    Args:
        n: number of simulations.
        method: phase difference calculation method.
        *kwargs: arguments for polarimeter_signal function.
    """
    results = []
    for i in range(n):
        xs, s1, s2 = polarimeter_signal(**kwargs)

        data = np.array([xs, s1, s2]).T
        data = pd.DataFrame(data=data, columns=["ANGLE", "A0", "A1"])
        data = data.groupby(['ANGLE'], as_index=False).agg({
            'A0': ['mean', 'std'],
            'A1': ['mean', 'std']
        })

        if len(data.index) == 1:
            raise ValueError("This is a file with only one angle!.")

        xs = np.array(data['ANGLE'])
        s1 = np.array(data['A0']['mean'])
        s2 = np.array(data['A1']['mean'])

        s1_sigma = np.array(data['A0']['std'])
        s2_sigma = np.array(data['A1']['std'])

        if np.isnan(s1_sigma).any() or (s1_sigma == 0).any():
            s1_sigma = None

        if np.isnan(s2_sigma).any() or (s2_sigma == 0).any():
            s2_sigma = None

        res = phase_difference(xs * 2, s1, s2, s1_sigma=s1_sigma, s2_sigma=s2_sigma, method=method)
        results.append(res)

    return results


def plot_two_signals(phi, folder, samples=1, s1_noise=None, s2_noise=None, show=False):
    print("")
    logger.info("TWO HARMONIC SIGNALS...")

    label_template = "(φ1 - φ2)={}°.\n(µ, σ1)={}\n(µ, σ1)={}"
    label = label_template.format(round(phi, 2), s1_noise, s2_noise).expandtabs(11)

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, folder=folder)

    plot.set_title(label)

    xs, s1, s2 = polarimeter_signal(
        A=1.7, cycles=1, phi=phi, fc=30, fa=samples, a0_noise=(0, 0.01), a1_noise=(0, 0.01),
        bits=16, max_v=ADC_MAXV, all_positive=True
    )

    plot.add_data(xs, s1, style='o-', color='k', xrad=True)
    plot.add_data(xs, s2, style='o-', xrad=True)
    plot._ax.set_ylim(0, ADC_MAXV)

    plot.save(filename='sim_two_signals-samples-{}'.format(samples))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def plot_methods(phi, folder, samples=5, step=1, max_cycles=10, reps=1, show=False):
    print("")
    logger.info("PHASE DIFFERENCE METHODS VS # OF CYCLES")

    cycles_list = np.arange(1, max_cycles + 1, step=1)

    title = "reps={}, samples={}, step={}".format(reps, samples, step)
    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES,
        title=title, ysci=True, xint=True,
        folder=folder
    )

    for method in PHASE_DIFFERENCE_METHODS:
        fc = samples_per_cycle(step=step)
        logger.info("Method: {}, fc={}, reps={}".format(method, fc, reps))

        errors = []
        for cycles in cycles_list:
            n_errors = n_simulations(
                n=reps, method=method, cycles=cycles, fc=fc, phi=phi,
                fa=samples, a0_noise=A0_NOISE, a1_noise=A1_NOISE, all_positive=True
            )

            # RMSE
            error_rad = np.sqrt(sum([abs(phi - e.value) ** 2 for e in n_errors]) / reps)
            error_degrees = np.rad2deg(error_rad)
            error_degrees_sci = "{:.2E}".format(error_degrees)

            errors.append(error_degrees)

            time = total_time(cycles) / 60
            logger.info("cycles={}, time={} m, φerr: {}.".format(cycles, time, error_degrees_sci))

        label = "{}".format(method)
        plot.add_data(cycles_list, errors, style='o-', lw=2, label=label)

    plot.legend(fontsize=12)

    plot.save(filename="sim_methods-reps-{}-samples-{}-step-{}.png".format(reps, samples, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def plot_error_vs_cycles(phi, folder, samples=5, max_cycles=10, reps=1, show=False):
    print("")
    logger.info("PHASE DIFFERENCE VS # OF CYCLES")

    cycles_list = np.arange(1, max_cycles + 1, step=1)

    title = "samples={}".format(samples)

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES, title=title, ysci=True, xint=True,
        folder=folder
    )

    n_reps = [1, 5, 10, 50]
    steps = [0.001, 0.01, 0.1, 1]
    for i, step in enumerate(steps, 0):
        fc = samples_per_cycle(step=step)
        reps = n_reps[i]
        logger.info("Method: {}, fc={}, reps={}".format('curve_fit', fc, reps))

        errors = []
        for cycles in cycles_list:
            n_res = n_simulations(
                n=reps, method='curve_fit', cycles=cycles, fc=fc, phi=phi,
                fa=samples, a0_noise=A0_NOISE, a1_noise=A1_NOISE, all_positive=True
            )

            # RMSE
            error_rad = np.sqrt(sum([abs(phi - e.value) ** 2 for e in n_res]) / reps)
            error_degrees = np.rad2deg(error_rad)
            error_degrees_sci = "{:.2E}".format(error_degrees)

            mean_u = sum([r.u for r in n_res]) / len(n_res)

            errors.append(error_degrees)

            time = total_time(cycles) / 60
            logger.info(
                "cycles={}, time={} m, φerr: {}, u: {}.".format(
                    cycles, time, error_degrees_sci, mean_u)
            )

        label = "step={}, reps={}".format(step, reps)
        plot.add_data(cycles_list, errors, style='.-', lw=2, label=label)

    plot.legend()

    plot.save(filename="sim_error_vs_cycles-samples-{}".format(samples))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def plot_error_vs_resolution(phi, folder, samples=5, step=1, max_cycles=10, reps=1, show=False):
    print("")
    logger.info("PHASE DIFFERENCE VS RESOLUTION")

    cycles_list = np.arange(1, max_cycles + 1, step=1)

    title = "step={}, samples={}, reps={}".format(step, samples, reps)

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES, title=title, ysci=True, xint=True,
        folder=folder
    )

    resolutions_bits = [(ARDUINO_BITS, ARDUINO_MAXV), (ADC_BITS, ADC_MAXV)]
    for bits, maxv in resolutions_bits:
        fc = samples_per_cycle(step=step)
        logger.info("fc={}".format(fc))

        amplitude = 0.9 * maxv

        errors = []
        for cycles in cycles_list:
            n_results = n_simulations(
                A=amplitude, bits=bits, max_v=maxv, n=reps, method='curve_fit', cycles=cycles,
                fc=fc, phi=phi, fa=samples, a0_noise=A0_NOISE, a1_noise=A1_NOISE, all_positive=True
            )

            # RMSE
            error_rad = np.sqrt(sum([abs(phi - res.value) ** 2 for res in n_results]) / reps)
            error_degrees = np.rad2deg(error_rad)
            error_degrees_sci = "{:.2E}".format(error_degrees)

            errors.append(error_degrees)

            time = total_time(cycles) / 60
            logger.info("cycles={}, time={} m, φerr: {}.".format(cycles, time, error_degrees_sci))

        label = "{} bits".format(bits)
        plot.add_data(cycles_list, errors, style='.-', lw=2, label=label)

    plot.legend(fontsize=12)

    filename = "sim_error_vs_resolution-step-{}-samples-{}-reps{}.png".format(step, samples, reps)
    plot.save(filename=filename)

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def plot_error_vs_range(phi, folder, samples=5, step=0.01, cycles=5, reps=1, show=False):
    print("")
    logger.info("PHASE DIFFERENCE VS MAX TENSION")

    title = "reps={}, samples={}, step={}".format(reps, samples, step)

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_MAX_V,
        title=title, ysci=True, xint=False,
        folder=folder
    )

    xs = np.arange(1, 1.7, step=0.1)

    errors = []
    for amplitude in xs:
        fc = samples_per_cycle(step=step)
        logger.info("A={}".format(amplitude))

        n_errors = n_simulations(
            A=amplitude, n=reps, method='curve_fit', cycles=cycles, fc=fc, phi=phi,
            fa=samples, a0_noise=A0_NOISE, a1_noise=A1_NOISE, all_positive=True
        )

        # RMSE
        error_rad = np.sqrt(sum([abs(phi - e.value) ** 2 for e in n_errors]) / reps)
        error_degrees = np.rad2deg(error_rad)
        error_degrees_sci = "{:.2E}".format(error_degrees)

        errors.append(error_degrees)

    time = total_time(cycles) / 60
    logger.info("cycles={}, time={} m, φerr: {}.".format(cycles, time, error_degrees_sci))

    plot.add_data(xs, errors, color='k', style='.-', lw=2)

    plot.save(
        filename="sim_error_vs_range-reps-{}-samples-{}-step-{}.png".format(reps, samples, step))

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
        cycles=cycles, fc=fc, phi=phi, fa=samples,
        a0_noise=A0_NOISE, a1_noise=A1_NOISE, all_positive=True
    )

    logger.info("Calculating phase difference...")
    res = phase_difference(xs * 2, s1, s2, method='curve_fit')

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

    plot.save(filename="sim_phase_diff_fit-samples-{}-step-{}.png".format(samples, step))

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
    if sim not in SIMULATIONS:
        raise ValueError("Simulation with name {} not implemented".format(sim))

    if sim in ['all', 'two_signals']:
        plot_two_signals(
            np.pi / 2, output_folder, samples, s1_noise=A0_NOISE, s2_noise=A1_NOISE, show=show)

    if sim in ['all', 'methods']:
        plot_methods(PHI, output_folder, samples, max_cycles=10, step=0.01, reps=reps, show=show)

    if sim in ['all', 'error_vs_cycles']:
        plot_error_vs_cycles(PHI, output_folder, samples, max_cycles=8, reps=reps, show=show)

    if sim in ['all', 'error_vs_res']:
        plot_error_vs_resolution(
            PHI, output_folder, samples, max_cycles=5, step=0.01, reps=reps, show=show)

    if sim in ['all', 'error_vs_range']:
        plot_error_vs_range(PHI, output_folder, samples, step=0.1, cycles=2, reps=reps, show=show)

    if sim in ['all', 'phase_diff']:
        plot_phase_diff(PHI, output_folder, samples, cycles=10, step=0.01, show=show)


if __name__ == '__main__':
    main(sim='error_vs_cycles', show=True)
