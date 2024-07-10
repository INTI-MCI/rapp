import numpy as np

from rapp.simulations import error_vs_resolution


np.random.seed(1)


def test_run(tmp_path):
    cycles, errors_per_bits = error_vs_resolution.run(
        angle=45, folder=tmp_path, reps=1, step=1, samples=50, cycles=1, save=False
    )

    print(errors_per_bits)
    assert len(cycles) == 1

    for errors in errors_per_bits.values():
        assert errors[-1] < 6e-2
