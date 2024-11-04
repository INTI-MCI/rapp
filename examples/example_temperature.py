import time
import logging

import serial
import numpy as np

from rapp.mocks import SerialMock

logger = logging.getLogger(__name__)

PORT = 'COM3'
BAUDRATE = 57600
TIMEOUT = 0.1
WAIT = 2
CMD = "temp?\n"
MEASUREMENT_WAIT = 60
MEASUREMENT_TIME = 3600

class ADCError(Exception):
    pass

@staticmethod
def get_serial_connection(*args, **kwargs):
    try:
        return serial.Serial(*args, **kwargs)
    except serial.serialutil.SerialException as e:
        raise ADCError("Error while making connection to serial port: {}".format(e))


def main(port=PORT, baudrate=BAUDRATE, timeout=TIMEOUT, wait=WAIT, cmd=CMD,
         measurement_wait=MEASUREMENT_WAIT, measurement_time=MEASUREMENT_TIME):
    print("Instantiating ADC...")
    serial_connection = get_serial_connection(port, baudrate=baudrate, timeout=timeout)
    logger.info("Waiting {} seconds after connecting to ADC...".format(wait))
    # Arduino resets when a new serial connection is made.
    # We need to wait, otherwise we don't receive anything.
    time.sleep(wait)

    hora = time.time()
    while time.time() - hora < measurement_time:
        hora_inicio = time.time()
        for i in range(10):
            serial_connection.write(bytes(cmd, 'utf-8'))
            time.sleep(1)
            temp = serial_connection.readline()
            # temp = float(temp.decode('utf-8').strip())
            print(temp)
        hora_fin = time.time()
        tiempo_mediciones = hora_fin - hora_inicio
        measurement_wait = measurement_wait - tiempo_mediciones
        time.sleep(measurement_wait)

if __name__ == '__main__':
    main()