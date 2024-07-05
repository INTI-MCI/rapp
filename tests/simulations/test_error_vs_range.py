from rapp.simulations import error_vs_range


# np.random.seed(1)

GAINS = {
    23: (6.144, 0.1875),
}


def test_run(tmp_path):
    percentages, errors_per_method = error_vs_range.run(
        angle=45,
        folder=tmp_path,
        step=1,
        samples=1,
        cycles=1,
        save=False,
    )

    print(errors_per_method)
    assert len(percentages) == 8

    for errors in errors_per_method.values():
        assert errors[-1] < 5e-2
