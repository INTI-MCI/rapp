import numpy as np

from rapp.simulations import error_vs_step


np.random.seed(0)


def test_run(tmp_path):
    steps, errors = error_vs_step.run(folder=tmp_path, mreps=[1], steps=[1], samples=1, save=False)

    for method, n_errors in errors.items():
        for e in n_errors:
            assert e < 6e-2
