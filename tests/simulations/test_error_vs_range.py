import numpy as np
from rapp.simulations import error_vs_range


np.random.seed(0)

GAINS = {
    23: (6.144, 0.1875),
}


def test_run(tmp_path):
    percentages, errors = error_vs_range.run(folder=tmp_path, samples=1, save=False)

    assert len(percentages) == 8

    for err in errors.values():
        assert err[-1] < 5e-2
