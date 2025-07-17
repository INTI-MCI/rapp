import time
# import pstats
time.sleep(1)
print("Hello")
time.sleep(1)
print("World")

# Para imprimir el contenido del archivo de salida del profiler:
# p = pstats.Stats('salida_profiler')
# p.sort_stats('cumulative').print_stats(10)

# Para correr el profiler con un script:
# python -m cProfile -o salida_profiler examples\\example_profiler.py