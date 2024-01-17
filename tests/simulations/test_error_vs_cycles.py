import numpy as np

from rapp.simulations import error_vs_cycles


np.random.seed(1)  # To make random simulations repeatable.


def test_run(tmp_path):
    cycles, errors_per_method = error_vs_cycles.run(
        phi=np.pi/4, folder=tmp_path,
        reps=1,
        step=1,
        samples=10,
        cycles=1,
        show=False,
        save=False
    )

    assert len(cycles) == 1
    for errors in errors_per_method.values():
        for error in errors:
            assert error < 1e-3

    std = np.std(list(errors_per_method.values()))
    assert std < 1e-4
