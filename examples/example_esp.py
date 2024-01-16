import time
from datetime import datetime  # noqa

from rapp.motion_controller import ESP301

controller = ESP301.build("COM3", 921600, axis=2, reset=False)

controller.set_velocity(4)
controller.set_acceleration(4)
controller.motor_on()


STEP = 4
reps = 5
print("PASO DE: {} grado(s).".format(STEP))

for rep in range(reps):
    controller.set_position(0)
    controller.set_position(STEP)

    values = []
    measure_time = 2
    start_time = time.time()
    while time.time() - start_time < measure_time:
        # print(datetime.now().isoformat(), measured_value)
        measured_value = controller.get_position()
        values.append(measured_value)

    print("Maximo error (rep {}): {}".format(rep + 1, abs(STEP - max(values))))

controller.close()
