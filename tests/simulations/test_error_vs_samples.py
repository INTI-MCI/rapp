import numpy as np

from rapp.simulations import error_vs_samples


np.random.seed(1)


def test_run(tmp_path):
    n_samples, errors = error_vs_samples.run(angle=45, folder=tmp_path, cycles=1, save=False)

    for method, m_errors in errors.items():
        for e in m_errors:
            assert e < 5e-2
