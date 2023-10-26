import time
from datetime import datetime

import esp

controller = esp.esp("COM3", 921600, 1, reset=False)

value = controller.setpos(0)
time.sleep(1)

controller.setpos(20)
for i in range(100):
    value = controller.getpos()
    print(datetime.now().isoformat(), value)
    time.sleep(0.01)

# file = open(filename, 'a')

controller.dev.close()
