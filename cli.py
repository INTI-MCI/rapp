import numpy as np

from rapp.plot import Plot
from rapp.utils import create_folder
from rapp.simulate import harmonic_signal

OUTPUT_FOLDER = 'output'


# Analyzer config. Step (degrees). Velocity (degrees per second).
STEP = 0.001
MAX_VELOCITY = 4
MIN_VELOCITY = STEP

# Measurement config.
MEASUREMENT_TIME = 300     # Seconds.
MEASUREMENT_VELOCITY = 4   # Degrees per second.
MEASUREMENT_STEP = 0.25    # How often do I take a measurement? In seconds.

# Plot phase diff vs number of cycles
MAX_N_CYCLES = 20          # Max number of cycles to test


def plot_simulations_test(phi, show=False):
    plot = Plot(ylabel="Voltage [V]", xlabel="Angle [rad]")
    plot.add_data(*harmonic_signal(), style='o-', label='φ=0', xrad=True)
    plot.add_data(*harmonic_signal(phi=-phi), style='o--', label=f'φ={round(phi, 2)}', xrad=True)

    plot.save(filename='two_signals_without_AWGN_noise')

    if show:
        plot.show()

    plot.clear()

    plot.add_data(*harmonic_signal(awgn=0.05), style='-', label='φ=0', xrad=True)
    plot.add_data(
        *harmonic_signal(phi=-phi, awgn=0.05), style='--', label=f'φ={round(phi, 2)}', xrad=True)

    plot.save(filename='two_signals_with_AWGN_noise')

    if show:
        plot.show()

    plot.close()


def plot_phase_diff_per_cycles(phi, show=False):

    print(f"MEASUREMENT_TIME: {MEASUREMENT_TIME} seconds.")
    print(f"MEASUREMENT_VELOCITY: {MEASUREMENT_VELOCITY} degrees per second.")
    print(f"MEASUREMENT_STEP: {MEASUREMENT_STEP} measurement per second.")

    total_samples = int((MEASUREMENT_TIME * MEASUREMENT_VELOCITY) / MEASUREMENT_STEP)
    cycles = np.arange(1, MAX_N_CYCLES + 1, step=1)

    errors = []
    for n_cycles in cycles:
        fs = int(total_samples / n_cycles)

        _, s1 = harmonic_signal(N=n_cycles, fs=fs)
        _, s2 = harmonic_signal(N=n_cycles, fs=fs, phi=-phi)

        phase_diff = np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))
        error = abs(phi - phase_diff)
        error_degrees = np.rad2deg(error)
        print(
            f"N={n_cycles}. fs={fs}. Error: {error_degrees}."
        )

        errors.append(error_degrees)

    min_error_arg = np.argmin(errors)
    print(min_error_arg)
    print(f"Minimum error was with {cycles[min_error_arg]} cycles.")

    title = "Cosine Similarity Test. \n Velocity={} deg/s. Time={} s. Step={}.".format(
        MEASUREMENT_VELOCITY, MEASUREMENT_TIME, MEASUREMENT_STEP
    )

    plot = Plot(ylabel="Error (°)", xlabel="N° of cycles", title=title)
    plot.add_data(cycles, errors, style='o-', color='k')
    # plot._ax.axvline(x=min_error_arg, ls='--', label=f"Min. error (x={min_error_arg})")

    plot.legend()

    plot.save(filename="phase_diff_per_cycles")

    if show:
        plot.show()


def main():
    create_folder(OUTPUT_FOLDER)

    phi = np.pi/4
    print(f"Phase difference: {np.rad2deg(phi)}")

    plot_simulations_test(phi=phi)

    plot_phase_diff_per_cycles(phi=phi, show=True)


if __name__ == '__main__':
    main()
