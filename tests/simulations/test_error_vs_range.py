import numpy as np

from rapp.simulations import error_vs_range


# np.random.seed(1)

GAINS = {
    23: (6.144, 0.1875),
}


def test_run(tmp_path):
    percentages, errors_per_maxv = error_vs_range.run(
        phi=np.pi / 4,
        folder=tmp_path,
        method='ODR',
        step=1,
        samples=1,
        cycles=1,
        save=False,
        gains=GAINS
    )

    print(errors_per_maxv)
    assert len(percentages) == 1

    for errors in errors_per_maxv.values():
        assert errors[-1] < 0.001
