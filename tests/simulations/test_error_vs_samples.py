import numpy as np

from rapp.simulations import error_vs_samples


def test_run(tmp_path):
    error_vs_samples.run(
        phi=np.pi/4, folder=tmp_path,
        method='ODR', reps=1, step=1, samples=1, cycles=1, show=False, save=False
    )
