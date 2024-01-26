
from rapp.simulations import one_phase_diff


def test_run(tmp_path):
    one_phase_diff.run(
        folder=tmp_path,
        angle=45,
        reps=1,
        step=1,
        samples=50,
        cycles=1,
        show=False,
        save=False
    )
