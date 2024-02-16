import numpy as np

from rapp.simulations import error_vs_step

np.random.seed(1)


def test_run(tmp_path):
    steps, errors = error_vs_step.run(
        angle=45,
        folder=tmp_path,
        reps=[1],
        step=[1],
        samples=1,
        cycles=1,
        save=False
    )

    assert errors[-1] < 5e-2
