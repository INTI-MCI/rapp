import numpy as np
import matplotlib.pyplot as plt

filepath1 = r'C:\Users\Admin\rapp\workdir\output-data\2024-11-07-1731004506.1177385-temperatura-tiempo-total-7200-tiempo-espera-120-muestras10.txt'
filepath2 = r'C:\Users\Admin\rapp\workdir\output-data\2024-11-07-mediciones-temp-sensores-calibrados.txt'

data1 = np.genfromtxt(filepath1, delimiter=',', skip_header=1)
promedio = np.mean(data1[:, :10], axis=1)
# Falta agregar lo que lee el último valor de cada fila para guardarlo como tiempos de medición

data2 = np.genfromtxt(filepath2, delimiter=',', skip_header=17, skip_footer=16)

plt.plot(promedio, label='promedio DS18B20')
plt.plot(data2[:, 0], label='sensor calibrado 1')
plt.plot(data2[:, 1], label='sensor calibrado 2')
plt.legend()
plt.show()
