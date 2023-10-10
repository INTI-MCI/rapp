import numpy as np

from rapp.plot import Plot
from rapp.utils import create_folder
from rapp.simulate import harmonic_signal

from matplotlib.ticker import MaxNLocator

OUTPUT_FOLDER = 'output'


# Analyzer velocity (degrees per second)
ANALYZER_VELOCITY = 4


def plot_harmonic_signal_test(phi, show=False):
    plot = Plot(ylabel="Voltage [V]", xlabel="Angle [rad]")
    plot.add_data(*harmonic_signal(), style='o-', label='φ=0', xrad=True)
    plot.add_data(*harmonic_signal(phi=-phi), style='o--', label=f'φ={round(phi, 2)}', xrad=True)

    plot.save(filename='two_signals_without_awgn_noise')

    if show:
        plot.show()

    plot.clear()

    plot.add_data(*harmonic_signal(awgn=0.05), style='-', label='φ=0', xrad=True)
    plot.add_data(
        *harmonic_signal(phi=-phi, awgn=0.05), style='--', label=f'φ={round(phi, 2)}', xrad=True)

    plot.save(filename='two_signals_with_awgn_noise')

    if show:
        plot.show()

    plot.close()


def plot_phase_diff_error_vs_cycles(phi, show=False):
    # Every how many degrees do I take a measurement?.
    step = 0.01

    # Max number of cycles to test
    max_n_cycles = 10

    print(f"ANALYZER_VELOCITY: {ANALYZER_VELOCITY} degrees per second.")
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


def plot_phase_diff_error_vs_step(phi, show=False):
    print(f"ANALYZER_VELOCITY: {ANALYZER_VELOCITY} degrees per second.")

    n_cycles = 10
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


def main():
    create_folder(OUTPUT_FOLDER)

    phi = np.pi/4
    print(f"Phase difference: {np.rad2deg(phi)}")

    plot_harmonic_signal_test(phi=phi)
    plot_phase_diff_error_vs_cycles(phi=phi)

    plot_phase_diff_error_vs_step(phi=phi)


if __name__ == '__main__':
    main()
