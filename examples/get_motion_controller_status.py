import serial
import time

MOTION_CONTROLLER_PORT = "COM4"
MOTION_CONTROLLER_BAUDRATE = 921600
ERROR_BUFFER_ESP301 = 1

GET_ERROR_BUFFER = True
SET_MOTORS_ON = False
RESET_CONTROLLER = False

serial_connection = serial.Serial(
    port=MOTION_CONTROLLER_PORT,
    baudrate=MOTION_CONTROLLER_BAUDRATE
)

if GET_ERROR_BUFFER:
    for i in range(ERROR_BUFFER_ESP301):
        serial_connection.write("TB?\r".encode())
        errmsg = serial_connection.readline()
        print(errmsg)
# b'113, 314883, AXIS-1 MOTOR NOT ENABLED\r\n'

if SET_MOTORS_ON:
    serial_connection.write("1MO;2MO;1MO?;2MO?\r".encode())
    print(serial_connection.readline())
    print(serial_connection.readline())

if RESET_CONTROLLER:
    serial_connection.write("RS\r".encode())
    serial_connection.close()
    time.sleep(30)
    serial_connection.open()
    serial_connection.write("TB?\r".encode())
    print(serial_connection.readline())

serial_connection.close()
