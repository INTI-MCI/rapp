import time
from datetime import datetime

from rapp import esp

controller = esp.ESP("COM3", 921600, axis=2, reset=False)

value = controller.setpos(10, axis=2)
controller.setvel(vel=4, axis=2)

"""

controller.setpos(20)
for i in range(100):
    value = controller.getpos()
    print(datetime.now().isoformat(), value)
    time.sleep(0.01)

# file = open(filename, 'a')
"""
controller.dev.close()
