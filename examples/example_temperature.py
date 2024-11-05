import time
import logging
import struct
import os
from datetime import date

import serial
import numpy as np
from pathlib import Path

from rapp.mocks import SerialMock
from rapp import constants as ct

logger = logging.getLogger(__name__)

PORT = 'COM3'
BAUDRATE = 57600
TIMEOUT = 0.1
WAIT = 2
CMD = "temp?\n"
MEASUREMENT_WAIT = 8#60
MEASUREMENT_TIME = 40#3600
SAMPLES = 2

FILENAME = 'temperatura'

params = "tiempo-total-{}-tiempo-espera-{}-muestras{}".format(MEASUREMENT_TIME, MEASUREMENT_WAIT, SAMPLES)
measurement_name = f"{date.today()}-{'temperatura'}-{params}.txt"
output_folder = r'C:\Users\Admin\rapp\workdir\output-data'
measurement_dir = os.path.join(output_folder, measurement_name)
# os.makedirs(measurement_dir, exist_ok=False)

class ADCError(Exception):
    pass

@staticmethod
def get_serial_connection(*args, **kwargs):
    try:
        return serial.Serial(*args, **kwargs)
    except serial.serialutil.SerialException as e:
        raise ADCError("Error while making connection to serial port: {}".format(e))


def main(port=PORT, baudrate=BAUDRATE, timeout=TIMEOUT, wait=WAIT, cmd=CMD,
         measurement_wait=MEASUREMENT_WAIT, measurement_time=MEASUREMENT_TIME, samples=SAMPLES):
    print("Instantiating ADC...")
    serial_connection = get_serial_connection(port, baudrate=baudrate, timeout=timeout)
    logger.info("Waiting {} seconds after connecting to ADC...".format(wait))
    # Arduino resets when a new serial connection is made.
    # We need to wait, otherwise we don't receive anything.
    time.sleep(wait)

    hora = time.time()
    while time.time() - hora < measurement_time:
        hora_inicio = time.time()
        temperaturas = [0] * samples
        for i in range(samples):
            serial_connection.write(bytes(cmd, 'utf-8'))
            time.sleep(0.5)
            temp = serial_connection.readline()
            temp = struct.unpack('<f', temp)[0]
            temperaturas[i] = '{}\n'.format(temp)
        print(temperaturas)
        with open(r"{measurement_dir}".format(measurement_dir=measurement_dir), 'a') as f:
            f.writelines(temperaturas)

        hora_fin = time.time()
        tiempo_mediciones = hora_fin - hora_inicio
        espera = measurement_wait - tiempo_mediciones
        time.sleep(espera)

if __name__ == '__main__':
    main()