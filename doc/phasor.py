import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from rapp.signal import simulator

dir_path = os.path.dirname(os.path.realpath(__file__))

phi = np.pi / 2

phi1 = np.pi / 2
phi2 = phi1 + phi / 2


xs, s1, s2 = simulator.polarimeter_signal(A=1, cycles=2, phi=phi, fc=360, fa=1, all_positive=True)
xs = xs
xs = xs * 2

r = np.arange(0, 1, 0.01)
theta1 = np.ones(len(r)) * phi1
theta2 = np.ones(len(r)) * phi2


fig = plt.figure(figsize=(8, 4))
ax1 = plt.subplot(121, projection='polar')
ax1.set_title("Analizador", y=0, pad=-25, verticalalignment="top", fontname='serif', fontsize=17)

ax1.set_rticks([])
ax1.plot(theta1, r, lw=2)
ax1.plot(theta2, r, lw=2)


ax2 = plt.subplot(122)

xs = xs / np.pi

ax2.plot(xs, s1, lw=2, color='C00')
ax2.plot(xs, s2, lw=2, color='C01')

ax2.plot(1, 2, 'o', color='C01', ms=7)
ax2.plot(1, 1, 'o', color='C00', ms=7)
ax2.set_xlim(0.5, 2.5)

ax2.axvline(x=1, ls='--', lw=2, color='k')
ax2.set_xlabel("Fase [rad]")
ax2.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\\pi$'))
ax2.xaxis.set_major_locator(tck.MultipleLocator(base=0.5))

fig.tight_layout()

fig.savefig(os.path.join(dir_path, "phasor.png"))

plt.show()

plt.close()
