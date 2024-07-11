import numpy as np

from rapp.simulations import error_vs_cycles


np.random.seed(0)


def test_run(tmp_path):
    cycles, errors = error_vs_cycles.run(folder=tmp_path, samples=10, cycles=1, save=False)

    assert len(cycles) == 2

    for err in errors.values():
        for e in err:
            assert e < 5e-2

    std = np.std(list(errors.values()))
    assert std < 5e-2
