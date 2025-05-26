import time
import logging
import struct
import os
from datetime import date
import json
import serial

''' 
This example provides the temperature measured by the temperature sensor used 
in polarimeter measurements (DS18B20 sensor).
'''

logger = logging.getLogger(__name__)

PORT = 'COM3'
BAUDRATE = 57600
TIMEOUT = 2
TIMEOUT_CONNECTION = 7  # Max time to wait for serial connection in seconds
MEASUREMENT_WAIT = 5  # Time we wait between temperature measurements in seconds
MEASUREMENT_TIME = 20  # Time of the measurement in seconds
SAMPLES = 1
CHANNEL = 1
CMD_REQ_TEMP = f"req-temp;{CHANNEL};\n"
CMD_COMPLETE = f"complete?;{CHANNEL};\n"
CMD_TEMP = f"temp;{CHANNEL};\n"

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


def wait_for_connection(serial_connection):
    start = time.time()
    elapsed_time = 0
    queries = 0
    while elapsed_time < TIMEOUT_CONNECTION:
        serial_connection.write(b'ready?\n')
        output = serial_connection.readline()
        end = time.time()
        elapsed_time = end - start
        if output == b'yes\r\n':
            line = serial_connection.readline()
            print("Data in input buffer after making connection: {}".format(line))
            serial_connection.reset_input_buffer()  # está un poco de más, lo dejo hasta saber mas de la comunicación
            break
        elif output == b'no\r\n':
            pass
        else:
            queries += 1

    if elapsed_time >= TIMEOUT_CONNECTION:
        raise ADCError(
            "Timeout while waiting for connection to open. Timeout = {} seconds with"
            " {} failed queries.".format(TIMEOUT_CONNECTION, queries)
        )
    print(
        "Connection opened in {} seconds with {} queries.".format(elapsed_time, queries)
    )


def main(port=PORT, baudrate=BAUDRATE, timeout=TIMEOUT, cmd_req_temp=CMD_REQ_TEMP, cmd_complete=CMD_COMPLETE,
         cmd_temp=CMD_TEMP, measurement_wait=MEASUREMENT_WAIT, measurement_time=MEASUREMENT_TIME, samples=SAMPLES):
    assert measurement_wait < measurement_time, "measurement_wait must be less than measurement_time"
    assert measurement_wait > samples * 0.5, "measurement_wait must be greater than samples * 0.5"

    print("Instantiating ADC...")
    serial_connection = get_serial_connection(port, baudrate=baudrate, timeout=timeout)
    wait_for_connection(serial_connection)

    flag_header = True
    hora = time.time()
    while time.time() - hora < measurement_time:
        temperaturas = [0] * samples
        hora_inicio = time.time()
        for i in range(samples):
            ask = True
            # serial_connection.reset_input_buffer()
            serial_connection.write(bytes(cmd_req_temp, 'utf-8'))

            while ask:
                serial_connection.write(bytes(cmd_complete, 'utf-8'))
                # print("ADC command: {}".format(cmd_complete))
                answer = serial_connection.read(1)
                ask = answer == b'\x00'
                # print("ADC response: {}".format(ask))
                if ask:
                    time.sleep(0.1)

            serial_connection.write(bytes(cmd_temp, 'utf-8'))
            temp = serial_connection.read(4)
            temp = struct.unpack('<f', temp)[0]
            temperaturas[i] = '{}'.format(temp)
        if flag_header:
            hora_fin = time.time()
            tiempo_mediciones = hora_fin - hora_inicio
            header = {'tiempo_total': measurement_time, 'tiempo_espera': measurement_wait, 'muestras': samples,
                      'tiempo_mediciones': tiempo_mediciones}
            header = json.dumps(header)
            with open(measurement_dir, 'w') as f:
                f.write(header + '\n')
            flag_header = False

        with open(measurement_dir, 'a') as f:
            hora_ = time.strftime("%H:%M:%S")
            f.writelines(",".join(temperaturas) + ',' + str(hora_) + '\n')
        print(temperaturas)

        hora_fin = time.time()
        tiempo_mediciones = hora_fin - hora_inicio
        print("Tiempo de mediciones: {}".format(tiempo_mediciones))
        espera = measurement_wait - tiempo_mediciones
        time.sleep(espera)


if __name__ == '__main__':
    main()
