import time
import pstats
time.sleep(1)
print("Hello")
time.sleep(1)
print("World")

# Para imprimir el contenido del archivo de salida del profiler:
p = pstats.Stats(r"C:\Users\Admin\rapp\workdir\output-data\salida_profiler_reps-20_acc-4_deac-4")
p.sort_stats('cumulative').print_stats(20)

# Para correr el profiler con un script:
# python -m cProfile -o salida_profiler examples\\example_profiler.py
