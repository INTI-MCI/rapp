import numpy as np

from rapp.simulations import error_vs_phase


np.random.seed(0)


def test_run(tmp_path):
    angles, errors = error_vs_phase.run(folder=tmp_path, samples=10, save=False)

    for err in errors.values():
        for e in err:
            assert e < 5e-2

    std = np.std(list(errors.values()))
    assert std < 5e-2
