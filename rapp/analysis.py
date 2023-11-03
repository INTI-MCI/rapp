import os

import numpy as np

from rapp.plot import Plot
from rapp.utils import create_folder
from rapp.simulate import harmonic_signal

from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt

from scipy import signal
from scipy.stats import norm
from scipy.optimize import curve_fit


OUTPUT_FOLDER = 'output'


ANALYZER_VELOCITY = 4  # Degrees per second.
PHI = np.pi/4


def cosine_similarity(s1, s2):
    return np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))


def phase_difference(xs, y1, y2):
    xs = xs * 2  # The cycle of the polarizer contains two cycles of the signal.

    popt1, pcov1 = curve_fit(sine, xs, y1)
    popt2, pcov2 = curve_fit(sine, xs, y2)

    fitx = np.arange(min(xs), max(xs), step=0.001)
    fity1 = sine(fitx, *popt1)
    fity2 = sine(fitx, *popt2)

    phi1 = popt1[1] % (np.pi)
    phi2 = popt2[1] % (np.pi)

    phase_diff = (phi1 - phi2) % (np.pi)
    phase_diff_degrees = np.rad2deg(phase_diff)

    fitx = fitx / 2  # The cycle of the polarizer contains two cycles of the signal.

    return phase_diff_degrees, fitx, fity1, fity2


def sine(x, a, phi, c):
    return a * np.sin(x + phi) + c


def samples_per_cycle(step=0.01):
    # Half cycle (180) of the analyzer is one full cycle of the signal.
    return int(180 / step)


def total_time(n_cycles):
    return n_cycles * (180 / ANALYZER_VELOCITY)


def pol1(x, A, B):
    return A * x + B


def pol2(x, A, B, C):
    return A * x**2 + B * x + C


def pol3(x, A, B, C, D):
    return A * x**3 + B * x**2 + C * x + D


def fit_and_plot(data, n_data, func, n):
    for i in np.arange(len(func)):
        popt, pcov = curve_fit(func[i], np.arange(data.size), data)
        plt.plot(np.arange(data.size), data)
        plt.plot(n_data, func[i](np.arange(data.size), *popt))
        plt.title(f'Ajuste deriva Canal {n}')
        plt.show(block=True)
        print(popt)
    return


def detrend_poly(data, func):
    popt, pcov = curve_fit(func, np.arange(data.size), data)
    # plt.plot(func(np.arange(data.size), *popt))
    # plt.plot(data)
    # plt.show()
    return data - func(np.arange(data.size), *popt)


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


def plot_sim_signals_and_phase_diff(phi, step=0.01, n_cycles=10, awgn=0.05, show=False):
    print("\n############ Phase difference error (sinusoidal fit):")

    fc = samples_per_cycle(step=step)

    xs, s1 = harmonic_signal(n=n_cycles, fc=fc, awgn=awgn, all_positive=True)
    _, s2 = harmonic_signal(n=n_cycles, fc=fc, phi=-phi, awgn=awgn, all_positive=True)

    phase_diff, fitx, fity1, fity2 = phase_difference(xs, s1, s2)

    print(f"Detected phase difference: {phase_diff}")

    error = abs(phi - phase_diff)
    error_degrees = np.rad2deg(error)

    print(f"n={n_cycles}, fc={fc}, step={step}, awgn={awgn}, φerr: {error_degrees}.")

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


def plot_signals_per_n_measurement(show=False):

    filenames = [
        'laser-75-int-alta.txt',
        'laser-75-encendido-15min.txt',
        'laser-16-reencendido-1M.txt',
        'laser-16-75-grados-int-baja.txt',
        'dark-current.txt'
    ]

    for filename in filenames:
        print(f"Graficando {filename}...")
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


def plot_signals_per_angle(show=False):
    filenames = [
        '2-full-cycles.txt',
        '2-full-cycles-2.txt',
        '1-full-cycles.txt',
        'test-clear-buffer.txt',
        'test-clear-buffer2.txt'
    ]

    for filename in filenames:
        filepath = os.path.join('data', filename)

        plot = Plot(ylabel="Voltage [V]", xlabel="Angle [rad]")
        plot.set_title(filename[:-4])

        cols = (0, 1, 2)
        data = np.loadtxt(filepath, delimiter=' ', skiprows=1, usecols=cols, encoding='iso-8859-1')
        voltage = data[:, 1]
        angles = data[:, 0] * np.pi / 180

        plot.add_data(angles, voltage, style='o-', color='k', xrad=True)
        plot.save(filename=filename[:-4])

        if show:
            plot.show()

        plot.close()


def plot_deriva():
    file = np.loadtxt(
        'data/laser-75-int-alta.txt',
        delimiter=' ', skiprows=1, usecols=(1, 2), encoding='iso-8859-1')

    ch0 = file[:, 0]
    ch1 = file[:, 1]

    reencendido = np.loadtxt(
        "data/laser-16-reencendido-1M.txt",
        delimiter=' ', skiprows=1, usecols=(1, 2), encoding='iso-8859-1')

    r0 = reencendido[:, 0]
    r1 = reencendido[:, 1]

    # fit_and_plot(r0[200000:], np.arange(r0[200000:].size), [pol1, pol2])
    # fit_and_plot(r1[200000:], np.arange(r1[200000:].size), [pol1, pol2])

    data_detrend0 = detrend_poly(r0[200000:], pol1)
    data_detrend1 = detrend_poly(r1[200000:], pol2)

    plt.figure()
    fft_data0 = np.fft.fft(data_detrend0)
    plt.plot(data_detrend0)
    fft_data1 = np.fft.fft(data_detrend1)
    plt.plot(data_detrend1)
    plt.show()
    plt.figure()
    plt.semilogy(np.abs(fft_data0))
    plt.semilogy(np.abs(fft_data1))
    plt.show()

    fit_and_plot(ch0, np.arange(ch0.size), [pol1], 0)
    fit_and_plot(ch1, np.arange(ch1.size), [pol2], 1)

    data_detrend0 = detrend_poly(ch0, pol1)
    data_detrend1 = detrend_poly(ch1, pol2)

    b, a = signal.butter(3, 0.064, btype='highpass')
    filtered_noise0 = signal.filtfilt(b, a, data_detrend0)
    filtered_noise1 = signal.filtfilt(b, a, data_detrend1)
    mu_0 = np.mean(filtered_noise0)
    std_dev0 = np.std(filtered_noise0)
    std_dev1 = np.std(filtered_noise1)
    mu_1 = np.mean(filtered_noise1)
    print('Media ch0 = ', mu_0)
    print('Sigma ch0 = ', std_dev0)
    print('Media ch1 = ', mu_1)
    print('Sigma ch1 = ', std_dev1)
    plt.figure()
    plt.title('Ruido filtrado')
    plt.plot(filtered_noise0, label='Canal 0')
    plt.plot(filtered_noise1, label='Canal 1')
    plt.legend()

    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Ruido filtrado')
    ax[0].hist(filtered_noise0, 100, range=(-0.005, 0.005))
    ax[0].set_title('Canal 0')
    ax[1].hist(filtered_noise1, 100, range=(-0.003, 0.003))
    ax[1].set_title('Canal 1')
    plt.show()

    # best fit of data
    (mu0, sigma0) = norm.fit(filtered_noise0)
    print('Media canal 0 = ', mu0)
    print('Sigma canal 0 = ', sigma0)

    (mu1, sigma1) = norm.fit(filtered_noise1)
    print('Media canal 1 = ', mu1)
    print('Sigma canal 1 = ', sigma1)

    # # add a 'best fit' line
    # plt.figure()
    # y = norm.pdf(np.linspace(min(filtered_noise0), max(filtered_noise0)), mu0, sigma0)
    # plt.hist(filtered_noise0, 100, range=(-0.005, 0.005), density=True)
    # plt.plot(np.linspace(-0.005, 0.005), y, 'r--', linewidth=2)
    # plt.show()


def plot_dark_current(show=False):

    filenames = [
        'dark-current.txt'
    ]

    for filename in filenames:
        print(f"Graficando {filename}...")
        filepath = os.path.join('data', filename)

        plot = Plot(ylabel="Voltage [V]", xlabel="# measurement")
        plot.set_title(filename[:-4])

        cols = (0, 1, 2)
        data = np.loadtxt(filepath, delimiter=' ', skiprows=1, usecols=cols, encoding='iso-8859-1')
        data = data[:, 1]
        xs = np.arange(1, data.size + 1, step=1)

        plot.add_data(xs, data, style='-', color='k')
        plot._ax.set_xlim(0, 500)

        plot.save(filename=f"{filename[:-4]}-signal")

        if show:
            plot.show()

        plot.close()

        plt.figure()
        plt.title("Dark current")
        plt.xlabel("Voltage")
        plt.ylabel("Count")
        plt.hist(data, bins='auto', color='k')
        plt.savefig(f"{os.path.join(OUTPUT_FOLDER, filename[:-4])}-histogram")

        if show:
            plot.show()

        plt.close()


def main():
    create_folder(OUTPUT_FOLDER)

    print(f"SIMULATED PHASE DIFFERENCE: {np.rad2deg(PHI)} degrees.")
    print(f"ANALYZER VELOCITY: {ANALYZER_VELOCITY} degrees per second.")

    # plot_harmonic_signals(phi=PHI)

    # plot_phase_diff_error_vs_cycles(phi=PHI, show=False)
    # plot_phase_diff_error_vs_step(phi=PHI, show=False)

    # plot_sim_signals_and_phase_diff(phi=PHI, n_cycles=20, step=0.01, awgn=0.01, show=False)

    # plot_signals_per_n_measurement(show=False)

    plot_deriva()

    # plot_signals_per_angle(show=False)

    # plot_dark_current(show=False)


if __name__ == '__main__':
    main()
