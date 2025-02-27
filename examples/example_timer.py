import schedule
import time


def callback():
    print('callback at: ', time.strftime("%H:%M:%S"))


schedule.every(5).seconds.do(callback)  # cada 5 segundos

inicio = time.time()
for i in range(2):
    schedule.run_pending()
    time.sleep(11)

final = time.time()
print(f"Tiempo transcurrido: {final - inicio} s.")