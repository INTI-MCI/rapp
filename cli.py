import numpy as np

from rapp.plot import Plot
from rapp.utils import create_folder
from rapp.simulate import simulate

OUTPUT_FOLDER = 'output'


def plot_simulations(phi):

    plot = Plot(ylabel="Voltage [V]", xlabel="Time [s]")
    plot.add_data(*simulate(t=1), style='-', color='k', label='φ=0')
    plot.add_data(
        *simulate(t=1, phi=phi), style='--', color='k', label=f'φ={round(phi, 2)}')

    plot.save(title='Two signals without noise')
    plot.show()

    plot.clear()

    plot.add_data(*simulate(t=1, awgn=0.1), style='-', color='k', label='φ=0')
    plot.add_data(
        *simulate(t=1, phi=phi, awgn=0.1), style='--', color='k', label=f'φ={round(phi, 2)}'
    )

    plot.save(title='Two signals with AWGN noise')
    plot.show()


def main():
    create_folder(OUTPUT_FOLDER)

    phi = -np.pi/4
    freq = 50
    # plot_simulations(phi=PHASE)

    print(f"True phase diff: {np.rad2deg(phi)}")

    seconds = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    errors = []

    cicles = []

    for t in seconds:
        ts, s1 = simulate(t=t, f=freq, phi=0)
        ts, s2 = simulate(t=t, f=freq, phi=phi)
        n_cicles = freq * t

        phase_diff = np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))
        error = phi + phase_diff
        print(
            f"Phase diff for {n_cicles} cicles: {np.rad2deg(phase_diff)}. "
            "Error: {np.rad2deg(error)}"
        )

        cicles.append(n_cicles)
        errors.append(error)

    plot = Plot(ylabel="Error (°)", xlabel="N° of cicles")
    plot.add_data(cicles, errors, style='o-', color='k')
    plot.save(title="Error of the phase difference")
    plot.show()


if __name__ == '__main__':
    main()
