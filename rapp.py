import os
import shutil

import numpy as np
from matplotlib import pyplot as plt

plt.style.use('style.mplstyle')

OUTPUT_FOLDER = 'output'


def create_folder(folder, overwrite=True):
    if os.path.exists(folder) and overwrite:
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)


def plot_signal(signal: tuple, title: str = '', show: bool = False) -> None:
    """Plots a signal and saves the figure.

    Args:
        signal: tuple with (x, y) data.
        title: title for the plot.
        show: if True, shows the plot.
    """

    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_ylabel("Voltage [V]")
    ax.set_xlabel("Time [s]")

    ax.plot(*signal, 'o', ms=4, mfc='None', color='k')

    create_folder(OUTPUT_FOLDER)
    filename = f'{title.lower().replace(" ", "_")}.png'
    fig.savefig(os.path.join(OUTPUT_FOLDER, filename))

    if show:
        plt.show()

    plt.close()


def simulate(A: float = 4, t: int = 1, n: int = 100, awgn: float = None) -> tuple:
    """Simulates a harmonic signal.

    Args:
        A: amplitude of the signal.
        t: number of periods.
        n: number of samples.
        awgn: amount of additive white gaussian noise, relative to A.

    Returns:
        The signal as an (x, y) tuple.
    """

    x = np.linspace(0, t * 2 * np.pi, num=n)

    signal = A * np.sin(x)

    noise = np.zeros(x.size)
    if awgn is not None:
        noise = np.random.normal(scale=A * awgn, size=x.size)

    signal = signal + noise

    return x, signal


def main():
    signal = simulate(t=3)
    plot_signal(signal, title='Signal without noise')

    awgn = 0.1
    signal_awgn = simulate(t=3, awgn=awgn)

    plot_signal(
        signal_awgn, title='Signal with additive white gaussian noise', show=True)


if __name__ == '__main__':
    main()
