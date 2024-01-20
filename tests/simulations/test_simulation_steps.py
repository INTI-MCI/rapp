from rapp.simulations import simulation_steps


def test_run(tmp_path):
    simulation_steps.run(folder=tmp_path, step=45, show=False, save=False)
