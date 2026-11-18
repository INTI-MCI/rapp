import numpy as np
import matplotlib.pyplot as plt

N_ORs = np.array([1, 2, 3, 4, 5])
ORs = np.array([11.6968547534413, 10.0025068793572, 9.82011361394627, 9.8293042363379, 9.84469619159776])
U_ORs = np.array([0.0268042027389904, 0.0383850239769632, 0.032, 0.0152339141056531, 0.0404085316092231])
OR_promedio = 9.829
U_RO_promedio = 0.013

Leyendas = ["L 1", "L 2", "10 repeticiones, paso 10 º", "10 repeticiones, paso 5 º", "5 repeticiones, paso 5 º"]
Marcadores = ["o", "o", "^", ".", "s"]


plt.figure()
# plt.errorbar(np.linspace(1, len(ORs[2:]), len(ORs[2:]), dtype=int), ORs[2:], yerr=U_ORs[2:], fmt=".", color="k", ecolor='0.5')

x = np.linspace(1, len(ORs[2:]), len(ORs[2:]), dtype=int)
for i, or_value in enumerate(ORs[2:]):
    plt.errorbar(i+1, [or_value], yerr=U_ORs[2:][i], fmt=".", color="0.1", ecolor='0.6', label=Leyendas[i+2], marker=Marcadores[i+2])

plt.hlines(9.8, 0.4, len(ORs[2:]) + 0.6, linestyles="dashed", color="b", alpha=0.5, label="Valor nominal")
plt.hlines(OR_promedio, 0.4, len(ORs[2:]) + 0.6, linestyles="dashed", color="r", alpha=0.5, label="Promedio ponderado")
plt.hlines(OR_promedio+U_RO_promedio, 0.4, len(ORs[2:]) + 0.6, linestyles="dashed", color="r", alpha=0.25, label="U promedio ponderado")
plt.hlines(OR_promedio-U_RO_promedio, 0.4, len(ORs[2:]) + 0.6, linestyles="dashed", color="r", alpha=0.25)

plt.xlim(0.4, len(ORs[2:]) + 0.55)
plt.ylabel("Rotación Óptica / °", size=14)
plt.xlabel("Nº de medición", size=14)
plt.xticks([1, 2, 3], fontsize=12)
plt.yticks(fontsize=12)
plt.title("Rotación Óptica a 20 °C", size=17)
plt.legend(loc="upper left", fontsize=14, ncol=2)
plt.tight_layout()
plt.show()

# agregar label de cada RO con paso y n repeticiones (ver que cada una esté con color o símbolo distinto)
# label del valor nominal y del promedio ponderado con su banda de error
# sacar decimales del eje x, que sólo muestre 1, 2 y 3
# ver si se puede sacar la barrita del ºC del título (probar con inkscape de última)
# tamaño labels y numeros en los ejes, ver que sea /ºC
# sacar comentarios antes de mandar a imprimir!!

