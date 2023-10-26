import os

import numpy as np

from rapp.plot import Plot
from rapp.utils import create_folder
from rapp.simulate import harmonic_signal

from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

OUTPUT_FOLDER = 'output'


ANALYZER_VELOCITY = 4  # Degrees per second.
PHI = np.pi/4


def cosine_similarity(s1, s2):
    return np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))


def sine(x, a, phi, c):
    return a * np.sin(x + phi) + c


def samples_per_cycle(step=0.01):
    # Half cycle (180) of the analyzer is one full cycle of the signal.
    return int(180 / step)


def total_time(n_cycles):
    return n_cycles * (180 / ANALYZER_VELOCITY)


def plot_harmonic_signals(phi, awgn=0.05, show=False):
    plot = Plot(ylabel="Voltage [V]", xlabel="Angle of analyzer [rad]")

    xs, s1 = harmonic_signal()
    _, s2 = harmonic_signal(phi=-phi)

    plot.add_data(xs, s1, color='k', style='o-', label='φ=0', xrad=True)
    plot.add_data(xs, s2, color='k', style='o-', label=f'φ={round(phi, 2)}', xrad=True)

    plot.save(filename='two_signals_without_awgn_noise')

    if show:
        plot.show()

    plot.clear()

    xs, s1 = harmonic_signal(awgn=awgn)
    _, s2 = harmonic_signal(phi=-phi, awgn=awgn)

    plot.add_data(xs, s1, color='k', style='o-', label='φ=0', xrad=True)
    plot.add_data(xs, s2, color='k', style='o-', label=f'φ={round(phi, 2)}', xrad=True)

    plot.save(filename='two_signals_with_awgn_noise')

    if show:
        plot.show()

    plot.close()


def plot_phase_diff_error_vs_cycles(phi, step=0.01, max_n_cycles=20, show=False):
    print("\n############ Phase difference error vs # cycles (cosine similarity):")
    fc = samples_per_cycle(step=step)
    cycles = np.arange(1, max_n_cycles + 1, step=1)

    errors = []
    for n_cycles in cycles:
        time = total_time(n_cycles)

        _, s1 = harmonic_signal(n=n_cycles, fc=fc)
        _, s2 = harmonic_signal(n=n_cycles, fc=fc, phi=-phi)

        phase_diff = cosine_similarity(s1, s2)

        error = abs(phi - phase_diff)
        error_degrees = np.rad2deg(error)

        error_degrees_sci = "{:.2E}".format(error_degrees)

        print(f"n={n_cycles}, fc={fc}, time={round(time/60, 1)} m, φerr: {error_degrees_sci}.")

        errors.append(error_degrees)

    label = f"fc={fc}. \nstep={step} deg."

    plot = Plot(ylabel="φ error (°)", xlabel="# cycles", title="Cosine Similarity", ysci=True)
    plot.add_data(cycles, errors, style='o-', color='k', label=label)
    plot._ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plot.legend()

    plot.save(filename="phase_diff_error_vs_cycles")

    if show:
        plot.show()

    plot.close()


def plot_phase_diff_error_vs_step(phi, n_cycles=20, show=False):
    print("\n############ Phase difference error vs step (cosine similarity):")
    time = total_time(n_cycles)

    steps = np.arange(0.1, 1.1, step=0.1)[::-1]

    errors = []
    for step in steps:
        fc = samples_per_cycle(step=step)

        _, s1 = harmonic_signal(n=n_cycles, fc=fc)
        _, s2 = harmonic_signal(n=n_cycles, fc=fc, phi=-phi)

        phase_diff = cosine_similarity(s1, s2)
        error_degrees = np.rad2deg(abs(phi - phase_diff))

        error_degrees_sci = "{:.2E}".format(error_degrees)

        print(f"n={n_cycles}, fc={fc}, step={round(step, 1)}, φerr: {error_degrees_sci}.")

        errors.append(error_degrees)

    label = f"# cycles={n_cycles}. \ntime={time/60} min."

    plot = Plot(
        ylabel="φ error (°)", xlabel="Step (degrees)", title="Cosine Similarity", ysci=True)

    plot.add_data(steps, errors, style='o-', color='k', label=label)

    plot.legend()

    plot.save(filename="phase_diff_error_vs_step")

    if show:
        plot.show()

    plot.close()


def plot_signals_and_phase_diff(phi, step=0.01, n_cycles=10, awgn=0.05, show=False):
    print("\n############ Phase difference error (sinusoidal fit):")

    fc = samples_per_cycle(step=step)

    xs, s1 = harmonic_signal(n=n_cycles, fc=fc, awgn=awgn, all_positive=True)
    _, s2 = harmonic_signal(n=n_cycles, fc=fc, phi=-phi, awgn=awgn, all_positive=True)

    xs = xs * 2  # The cycle of the polarizer contains two cycles of the signal.

    popt1, pcov1 = curve_fit(sine, xs, s1)
    popt2, pcov2 = curve_fit(sine, xs, s2)

    fitx = np.arange(min(xs), max(xs), step=0.001)
    fity1 = sine(fitx, *popt1)
    fity2 = sine(fitx, *popt2)

    phi1 = popt1[1] % (np.pi)
    phi2 = popt2[1] % (np.pi)

    phase_diff = (phi1 - phi2) % (np.pi)
    phase_diff_degrees = np.rad2deg(phase_diff)

    print(f"Detected phase difference: {phase_diff_degrees}")

    error = abs(phi - phase_diff)
    error_degrees = np.rad2deg(error)

    print(f"n={n_cycles}, fc={fc}, step={step}, awgn={awgn}, φerr: {error_degrees}.")

    # Go back to the xs angles of the polarizer.
    xs = xs / 2
    fitx = fitx / 2

    label = (
        f"fc={fc}. \n"
        f"step={step} deg. \n"
        f"# cycles={n_cycles}. \n"
        f"|φ1 - φ2| = {round(np.rad2deg(phi))}°. \n"
        f"AWGN={awgn}"
    )

    plot = Plot(ylabel="Voltage [V]", xlabel="Angle of analyzer [rad]")

    markevery = int(fc / 180) if fc >= 180 else 1  # for visualization purposes, show less points.

    plot.add_data(
        xs, s1,
        ms=6, color='k', mew=0.5, xrad=True, markevery=markevery, alpha=0.8, label=label
    )

    plot.add_data(
        xs, s2,
        ms=6, color='k', mew=0.5, xrad=True, markevery=markevery, alpha=0.8
    )

    label = f"φerr = {round(error_degrees, 5)}."
    plot.add_data(fitx, fity1, style='-', color='k', lw=1.5, xrad=True)
    plot.add_data(fitx, fity2, style='-', color='k', lw=1.5, xrad=True, label=label)

    plot._ax.set_xlim(0, 1)

    plot.legend(loc='upper right')

    plot.save(filename="phase_diff_sine_fit")

    if show:
        plot.show()

    plot.close()


def plot_noise(show=False):

    filenames = [
        'laser-75-int-alta.txt',
        'laser-75-encendido-15min.txt',
        'laser-16-reencendido-1M.txt',
        'laser-16-75-grados-int-baja.txt',
        'dark-current.txt'
    ]

    for filename in filenames:
        filepath = os.path.join('data', filename)

        plot = Plot(ylabel="Voltage [V]", xlabel="# measurement")
        plot.set_title(filename[:-4])

        cols = (0, 1, 2)
        data = np.loadtxt(filepath, delimiter=' ', skiprows=1, usecols=cols, encoding='iso-8859-1')
        data = data[:, 1]
        xs = np.arange(1, data.size + 1, step=1)

        plot.add_data(xs, data, style='-', color='k')
        plot.save(filename=filename[:-4])

        if show:
            plot.show()

        plot.close()


def main():
    create_folder(OUTPUT_FOLDER)

    print(f"SIMULATED PHASE DIFFERENCE: {np.rad2deg(PHI)} degrees.")
    print(f"ANALYZER VELOCITY: {ANALYZER_VELOCITY} degrees per second.")

    # plot_harmonic_signals(phi=PHI)

    # plot_phase_diff_error_vs_cycles(phi=PHI, show=False)
    # plot_phase_diff_error_vs_step(phi=PHI, show=False)

    # plot_signals_and_phase_diff(phi=PHI, n_cycles=20, step=0.01, awgn=0.01, show=False)

    plot_noise(show=False)


if __name__ == '__main__':
    main()
