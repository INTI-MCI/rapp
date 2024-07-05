import numpy as np

from rapp.simulations import error_vs_step

np.random.seed(1)


def test_run(tmp_path):
    steps, errors = error_vs_step.run(
        angle=45,
        folder=tmp_path,
        reps=1,
        steps=[1],
        samples=1,
        cycles=1,
        save=False
    )

    for method, m_errors in errors.items():
        for e in m_errors:
            assert e < 6e-2
