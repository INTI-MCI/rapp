import numpy as np

from rapp.simulations import error_vs_phase


np.random.seed(1)  # To make random simulations repeatable.


def test_run(tmp_path):
    angles, errors_per_method = error_vs_phase.run(
        folder=tmp_path,
        reps=1,
        step=1,
        samples=10,
        cycles=1,
        show=False,
        save=False
    )

    for errors in errors_per_method.values():
        for error in errors:
            assert error < 5e-2

    std = np.std(list(errors_per_method.values()))
    assert std < 5e-2
