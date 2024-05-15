import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest

print("\nThis example has two parts:\n"
      " 1) show in a plot that the RMSE of an averaging operation is higher than the theoretical\n"
      "    std(mean) when the std of the added Gaussian noise is less that half the quantization\n"
      "    quantum.\n"
      " 2) show that the random variable given by mean(array of Poisson noise) has a Gaussian\n"
      "    distribution.")

n_samples = 100
n_stds = 25
n_bits = 16
n_reps = 50
x_value = 2.3

quantum = 4.028 / 2**n_bits
stds = np.logspace(-2, 2, n_stds) * quantum
rmse_non_quant = stds / np.sqrt(n_samples)

rmse = np.zeros(n_stds)
for k, std in enumerate(stds):
    errors = np.zeros(n_reps)
    for krep in range(n_reps):
        x_real = np.random.normal(loc=x_value, scale=std, size=n_samples)
        x_quant = np.round(x_real / quantum) * quantum
        errors[krep] = np.mean(x_quant) - x_value

    rmse[k] = np.sqrt(np.mean(errors ** 2))

plt.loglog(stds / quantum, rmse / rmse_non_quant)
plt.xlabel("Std of noise / quantum")
plt.ylabel("RMSE / RMSE_non_quantized(theoretical)")

n_samples_draw = 1000
lam = 1000

gaussian_intensities = np.random.normal(lam, lam, size=n_samples_draw)
print(f"Test statistics using {n_samples_draw} samples:")
print("\nScores for a normal distribution")
stats, pvalue = normaltest(gaussian_intensities)
print(stats)
print(pvalue)

intensities = np.random.poisson(lam=lam, size=n_samples_draw)
print(f"\nDrawing {n_samples_draw} Poisson samples with lambda={lam}.")
print(f"Mean: {np.mean(intensities):.3f}, std: {np.std(intensities):.3f}")
# fig = plt.figure()
# plt.hist(intensities, bins=25)
# plt.show()

print("Scores for a Poisson distribution")
stats, pvalue = normaltest(intensities)
print(stats)
print(pvalue)

print(f"\nScores for the distribution of the mean value of {n_samples} Poisson variables")
mean_intensities = np.array([np.mean(np.random.poisson(lam=lam, size=n_samples))
                             for i in range(n_samples_draw)])
stats, pvalue = normaltest(mean_intensities)
print(stats)
print(pvalue)

plt.show()
