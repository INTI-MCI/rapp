import numpy as np

from rapp.plot import Plot
from rapp.utils import create_folder
from rapp.simulate import simulate

OUTPUT_FOLDER = 'output'


def main():
    create_folder(OUTPUT_FOLDER)

    PHASE = -np.pi/4

    plot = Plot(ylabel="Voltage [V]", xlabel="Time [s]")
    plot.add_data(*simulate(t=2), style='-', color='k', label='φ=0')
    plot.add_data(
        *simulate(t=2, phi=PHASE), style='--', color='k', label=f'φ={round(PHASE, 2)}')

    plot.save(title='Two signals without noise')
    plot.show()

    plot.clear()

    plot.add_data(*simulate(t=2, awgn=0.1), style='-', color='k', label='φ=0')
    plot.add_data(
        *simulate(t=2, phi=PHASE, awgn=0.1), style='--', color='k', label=f'φ={round(PHASE, 2)}'
    )

    plot.save(title='Two signals with AWGN noise')
    plot.show()


if __name__ == '__main__':
    main()
