import numpy as np

from rapp.simulations import one_phase_diff


def test_run(tmp_path):
    one_phase_diff.run(
        phi=np.pi/4, folder=tmp_path,
        method='ODR', reps=1, step=1, samples=5, cycles=1, show=False, save=False
    )
