import numpy as np

from rapp.simulations import pvalue_vs_range


np.random.seed(0)


def test_run(tmp_path):
    xs, mean_pvalues = pvalue_vs_range.run(folder=tmp_path, save=False)

    print(mean_pvalues)
    for i, pvalue in enumerate(mean_pvalues):
        if i > 6:
            assert pvalue > 0.05
