import time
from datetime import datetime

from rapp import esp

controller = esp.ESP("COM3", 921600, axis=2, reset=False)

controller.setvel(vel=4)

controller.setpos(0)
ref_value = controller.setpos(4.5)

print("valor referencia: {}".format(ref_value))

measure_time = 3
start_time = time.time()
while time.time() - start_time < measure_time:
    measured_value = controller.getpos()
    print(datetime.now().isoformat(), measured_value)
    time.sleep(0.01)

controller.dev.close()
