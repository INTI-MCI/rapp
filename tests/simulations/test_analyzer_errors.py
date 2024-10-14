import numpy as np

from rapp.simulations import sim_analyzer_errors


np.random.seed(0)


def test_run(tmp_path):
    angle_props_lists, errors = sim_analyzer_errors.run(folder=tmp_path, cycles=1, save=False)

    for m_errors in errors.values():
        assert np.all(m_errors < 0.2)
