import time
import logging
import struct
import os
from datetime import date
import json
import serial


logger = logging.getLogger(__name__)

PORT = 'COM3'
BAUDRATE = 57600
TIMEOUT = 1
WAIT = 2
CMD = "temp?\n"
MEASUREMENT_WAIT = 7#60
MEASUREMENT_TIME = 30#3600
SAMPLES = 10

FILENAME = 'temperatura'

params = "tiempo-total-{}-tiempo-espera-{}-muestras{}".format(MEASUREMENT_TIME, MEASUREMENT_WAIT, SAMPLES)
measurement_name = f"{date.today()}-{time.time()}-{'temperatura'}-{params}.txt"
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
    assert measurement_wait < measurement_time, "measurement_wait must be less than measurement_time"
    assert measurement_wait > samples * 0.5, "measurement_wait must be greater than samples * 0.5"

    print("Instantiating ADC...")
    serial_connection = get_serial_connection(port, baudrate=baudrate, timeout=timeout)
    logger.info("Waiting {} seconds after connecting to ADC...".format(wait))
    # Arduino resets when a new serial connection is made.
    # We need to wait, otherwise we don't receive anything.
    time.sleep(wait)
    flag_header = True
    hora = time.time()
    while time.time() - hora < measurement_time:
        temperaturas = [0] * samples
        hora_inicio = time.time()
        for i in range(samples):
            serial_connection.write(bytes(cmd, 'utf-8'))
            temp = serial_connection.read(4)
            temp = struct.unpack('<f', temp)[0]
            temperaturas[i] = '{}'.format(temp)
        if flag_header:
            hora_fin = time.time()
            tiempo_mediciones = hora_fin - hora_inicio
            header = {'tiempo_total': measurement_time, 'tiempo_espera': measurement_wait, 'muestras': samples, 'tiempo_mediciones': tiempo_mediciones}
            header = json.dumps(header)
            with open(measurement_dir, 'w') as f:
                f.write(header + '\n')
            flag_header = False

        with open(measurement_dir, 'a') as f:
            f.writelines(",".join(temperaturas) + '\n')
        print(temperaturas)

        hora_fin = time.time()
        tiempo_mediciones = hora_fin - hora_inicio
        espera = measurement_wait - tiempo_mediciones
        time.sleep(espera)

if __name__ == '__main__':
    main()