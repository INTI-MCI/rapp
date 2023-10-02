import numpy as np

from rapp.plot import Plot
from rapp.utils import create_folder
from rapp.simulate import simulate

OUTPUT_FOLDER = 'output'


def main():
    create_folder(OUTPUT_FOLDER)

    PHASE = -np.pi/4

    plot = Plot(ylabel="Voltage [V]", xlabel="Time [s]")
    plot.add_data(*simulate(t=1), style='-', color='k', label='φ=0')
    plot.add_data(
        *simulate(t=1, phi=PHASE), style='--', color='k', label=f'φ={round(PHASE, 2)}')

    plot.save(title='Two signals without noise')
    plot.show()

    plot.clear()

    plot.add_data(*simulate(t=1, awgn=0.1), style='-', color='k', label='φ=0')
    plot.add_data(
        *simulate(t=1, phi=PHASE, awgn=0.1), style='--', color='k', label=f'φ={round(PHASE, 2)}'
    )

    plot.save(title='Two signals with AWGN noise')
    plot.show()

    print(f"True phase diff: {np.rad2deg(PHASE)}")
    ts, s1 = simulate(t=5, phi=0)
    ts, s2 = simulate(t=5, phi=PHASE)

    phase_diff = np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))
    error = PHASE + phase_diff
    print(f"Phase diff (cosine similarity): {np.rad2deg(phase_diff)}. Error: {np.rad2deg(error)}")


if __name__ == '__main__':
    main()
