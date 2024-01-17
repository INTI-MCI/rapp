import numpy as np

from rapp.simulations import error_vs_samples


np.random.seed(1)


def test_run(tmp_path):
    n_samples, errors = error_vs_samples.run(phi=np.pi / 4, folder=tmp_path, cycles=1, save=False)

    for error in errors:
        assert error < 0.0003
