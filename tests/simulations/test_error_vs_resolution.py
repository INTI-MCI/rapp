import numpy as np

from rapp.simulations import error_vs_resolution


np.random.seed(0)


def test_run(tmp_path):
    cycles, errors_per_bits = error_vs_resolution.run(
        folder=tmp_path, samples=50, cycles=1, save=False
    )

    print(errors_per_bits)
    assert len(cycles) == 2

    for errors in errors_per_bits.values():
        print(errors)
        # assert errors[-1] < 6e-2
