import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from rapp.signal import simulator

dir_path = os.path.dirname(os.path.realpath(__file__))

phi = np.pi / 2

phi1 = np.pi / 2
phi2 = phi1 - phi / 2

ANALYZER_AXIS = 0.5
COLOR_REFERENCE = 'C00'
COLOR_OBJECT = 'C01'

xs0, s1, s2 = simulator.polarimeter_signal(A=1, cycles=3, phi=-phi, fc=90, fa=1, all_positive=True)

# Set analyzer axis
xs = xs0 - 3 * np.pi / 4

r = np.arange(0, 1, 0.01)
theta1 = np.ones(len(r)) * phi1
theta2 = np.ones(len(r)) * phi2


fig = plt.figure(figsize=(8, 4.7))
ax1 = plt.subplot(121, projection='polar')
ax1.set_title("Analizador", y=0, pad=-25, verticalalignment="top", fontname='serif', fontsize=17)

ax1.set_rticks([])
ax1.plot(theta1, r, lw=2, color=COLOR_REFERENCE, label="Plano del haz de referencia")
ax1.plot(theta1 + np.pi, r, lw=2, color=COLOR_REFERENCE)
ax1.plot(theta2, r, lw=2, color=COLOR_OBJECT, label="Plano del haz objeto")
ax1.plot(theta2 + np.pi, r, lw=2, color=COLOR_OBJECT)
ax1.plot(theta1 * 0, r, lw=2, color="k",  ls="--", label="Eje del analizador")
ax1.plot(theta1 * 0 + np.pi, r,  lw=2, color="k", ls="--")
ax1.legend(bbox_to_anchor=(0, 1.1), loc="lower left")


xs = xs / np.pi
ax2 = plt.subplot(122)
ax2.set_xlabel("Posición del Analizador [rad]")

ax2.plot(xs, s1, 'o', lw=2, color=COLOR_REFERENCE,  mfc='None', label="referencia")
ax2.plot(xs, s2, 'o', lw=2, color=COLOR_OBJECT,  mfc='None', label="objeto")
ax2.legend(loc='upper right', fontsize=10)

ax2.set_xlim(0, 2)
ax2.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\\pi$'))
ax2.xaxis.set_major_locator(tck.MultipleLocator(base=0.5))
ax2.xaxis.set_major_locator(plt.MaxNLocator(4))

ax3 = ax2.twiny()
ax3.plot(xs0 / np.pi, np.ones(len(xs)), alpha=0)
ax3.set_xlim(0, 4)
ax3.set_xlabel("Fase de la señal objeto")
ax3.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\\pi$'))
ax3.xaxis.set_major_locator(tck.MultipleLocator(base=0.5))
ax3.xaxis.set_major_locator(plt.MaxNLocator(4))

fig.tight_layout()

fig.savefig(os.path.join(dir_path, "phasor.png"))

plt.show()

plt.close()
