import numpy as np

from rapp.plot import Plot
from rapp.utils import create_folder
from rapp.simulate import harmonic_signal

from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

OUTPUT_FOLDER = 'output'


ANALYZER_VELOCITY = 4  # Degrees per second.


def sine(x, a, phi):
    return a * np.sin(x + phi)


def plot_harmonic_signals(phi, awgn=0.05, show=False):
    plot = Plot(ylabel="Voltage [V]", xlabel="Angle [rad]")

    xs, s1 = harmonic_signal()
    _, s2 = harmonic_signal(phi=-phi)

    plot.add_data(xs, s1, color='k', label='φ=0', xrad=True)
    plot.add_data(xs, s2, color='k', label=f'φ={round(phi, 2)}', xrad=True)

    plot.save(filename='two_signals_without_awgn_noise')

    if show:
        plot.show()

    plot.clear()

    xs, s1 = harmonic_signal(awgn=awgn)
    _, s2 = harmonic_signal(phi=-phi, awgn=awgn)

    plot.add_data(xs, s1, color='k', label='φ=0', xrad=True)
    plot.add_data(xs, s2, color='k', label=f'φ={round(phi, 2)}', xrad=True)

    plot.save(filename='two_signals_with_awgn_noise')

    if show:
        plot.show()

    plot.close()


def plot_phase_diff_error_vs_cycles(phi, step=0.01, show=False):
    max_n_cycles = 10  # Max number of cycles to test

    print(f"MEASUREMENT_STEP: 1 measurement for each {step} degree step.")

    cycles = np.arange(1, max_n_cycles + 1, step=1)
    fs = int(360 / step)

    errors = []
    for n_cycles in cycles:
        total_time = n_cycles * (360 / ANALYZER_VELOCITY)

        _, s1 = harmonic_signal(N=n_cycles, fs=fs)
        _, s2 = harmonic_signal(N=n_cycles, fs=fs, phi=-phi)

        phase_diff = np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))

        error = abs(phi - phase_diff)
        error_degrees = np.rad2deg(error)

        print(f"N={n_cycles}. fs={fs}. time={total_time/60} m. Error: {error_degrees}.")

        errors.append(error_degrees)

    title = (
        "Cosine Similarity Test. \n"
        f"Velocity={ANALYZER_VELOCITY} deg/s. Fs={fs}. Step={step} deg."
    )

    plot = Plot(ylabel="Error (°)", xlabel="N° of cycles", title=title)
    plot.add_data(cycles, errors, style='o-', color='k')
    plot._ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plot.legend()

    plot.save(filename="phase_diff_error_vs_cycles")

    if show:
        plot.show()

    plot.close()


def plot_phase_diff_error_vs_step(phi, n_cycles=10, show=False):

    total_time = n_cycles * (360 / ANALYZER_VELOCITY)

    steps = np.arange(0.1, 1.1, step=0.1)[::-1]

    errors = []
    for step in steps:
        fs = int(360 / step)

        _, s1 = harmonic_signal(N=n_cycles, fs=fs)
        _, s2 = harmonic_signal(N=n_cycles, fs=fs, phi=-phi)

        phase_diff = np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))
        error_degrees = np.rad2deg(abs(phi - phase_diff))

        print(f"N={n_cycles}. fs={fs}. time={total_time/60} m. Error: {error_degrees}.")

        errors.append(error_degrees)

    title = (
        "Cosine Similarity Test. \n"
        f"Velocity={ANALYZER_VELOCITY} deg/s. Time={total_time/60} min."
    )

    plot = Plot(ylabel="Error (°)", xlabel="Step (degrees)", title=title)
    plot.add_data(steps, errors, style='o-', color='k')

    plot.legend()

    plot.save(filename="phase_diff_error_vs_step")

    if show:
        plot.show()

    plot.close()


def plot_signals_and_phase_diff(phi, step=0.01, n_cycles=10, awgn=0.05, show=False):
    print(f"MEASUREMENT STEP: {step} degrees.")

    fs = int(360 / step)

    total_time = n_cycles * (360 / ANALYZER_VELOCITY)

    xs, s1 = harmonic_signal(N=n_cycles, fs=fs, awgn=awgn)
    _, s2 = harmonic_signal(N=n_cycles, fs=fs, phi=-phi, awgn=awgn)

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

    print(f"N={n_cycles}. fs={fs}. time={total_time/60} m. Error: {error_degrees}.")

    title = (
        "Sine Fit Test. \n"
        f"Velocity={ANALYZER_VELOCITY} deg/s. Fs={fs}. \n"
        f"Step={step} deg. Time={total_time/60} min. \n"
        f"φ = |φ1 - φ2| = {round(np.rad2deg(phi))}°."
    )

    plot = Plot(ylabel="Voltage [V]", xlabel="Angle [rad]", title=title)

    plot.add_data(
        xs, s1,
        ms=6, color='k', mew=0.5, xrad=True, markevery=200, alpha=0.8,
        label=f'AWGN={awgn}'
    )

    plot.add_data(
        xs, s2,
        ms=6, color='k', mew=0.5, xrad=True, markevery=200,
        alpha=0.8
    )

    label = f"|φ' - φ|  = {round(error_degrees, 7)}"
    plot.add_data(fitx, fity1, style='-', color='k', lw=1.5, xrad=True)
    plot.add_data(fitx, fity2, style='-', color='k', lw=1.5, xrad=True, label=label)

    plot._ax.set_xlim(0, 2)

    plot.legend(loc='upper right')

    plot.save(filename="phase_diff_sine_fit")

    if show:
        plot.show()

    plot.close()


def main():
    create_folder(OUTPUT_FOLDER)

    phi = np.pi/4
    print(f"SIMULATED PHASE DIFFERENCE: {np.rad2deg(phi)} degrees.")
    print(f"ANALYZER VELOCITY: {ANALYZER_VELOCITY} degrees per second.")

    plot_harmonic_signals(phi=phi)

    plot_phase_diff_error_vs_cycles(phi=phi)
    plot_phase_diff_error_vs_step(phi=phi)

    plot_signals_and_phase_diff(phi=phi, n_cycles=10, step=0.01, awgn=0.01, show=True)


if __name__ == '__main__':
    main()
