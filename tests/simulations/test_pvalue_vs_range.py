import numpy as np

from rapp.simulations import pvalue_vs_range


np.random.seed(1)


def test_run(tmp_path):
    xs, mean_pvalues = pvalue_vs_range.run(np.pi / 4, tmp_path, save=False)

    print(mean_pvalues)
    for i, pvalue in enumerate(mean_pvalues):
        if i < 5:
            assert np.isnan(pvalue) or pvalue < 0.05
        else:
            assert pvalue > 0.05
