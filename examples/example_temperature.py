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
MEASUREMENT_WAIT = 10  # 120  # Time we wait between temperature measurements in seconds
MEASUREMENT_TIME = 30  # 50400  # Time of the measurement in seconds
SAMPLES = 10
CMD_REQ_TEMP = "req-temp;{};\n"
CMD_COMPLETE = "complete?;{};\n"
CMD_TEMP = "temp;{};\n"

FILENAME = 'temperatura'

# output_folder = r'C:\Users\cvargas\rapp\workdir\output-data' # Para la compu portatil del labo
output_folder = r"C:\Users\Admin\rapp\workdir\output-data"
sub_folder = r"{d}-{t}-{filename}".format(d=date.today(), t=time.time(), filename=FILENAME)
measurement_dir = os.path.join(output_folder, sub_folder)
os.makedirs(measurement_dir, exist_ok=False)
params = r"tiempo-total-{}-tiempo-espera-{}-muestras-{}".format(MEASUREMENT_TIME, MEASUREMENT_WAIT, SAMPLES)
measurement_name_0 = f"sensor-0-{params}.txt"
measurement_path_0 = os.path.join(measurement_dir, measurement_name_0)
measurement_name_1 = f"sensor-1-{params}.txt"
measurement_path_1 = os.path.join(measurement_dir, measurement_name_1)


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
            # print("Data in input buffer after making connection: {}".format(line))
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
    request_0 = cmd_req_temp.format(0)
    request_1 = cmd_req_temp.format(1)
    complete_0 = cmd_complete.format(0)
    complete_1 = cmd_complete.format(1)
    read_0 = cmd_temp.format(0)
    read_1 = cmd_temp.format(1)
    while time.time() - hora < measurement_time:
        temperaturas0 = [0] * samples
        temperaturas1 = [0] * samples
        hora_inicio = time.time()
        for i in range(samples):
            ask1 = True
            ask2 = True
            # serial_connection.reset_input_buffer()
            serial_connection.write(bytes(request_0, 'utf-8'))
            serial_connection.write(bytes(request_1, 'utf-8'))

            while ask1 or ask2:
                serial_connection.write(bytes(complete_0, 'utf-8'))
                answer1 = serial_connection.read(1)
                ask1 = answer1 == b'\x00'

                serial_connection.write(bytes(complete_1, 'utf-8'))
                answer2 = serial_connection.read(1)
                ask2 = answer2 == b'\x00'
                if ask1 or ask2:
                    time.sleep(0.1)

            serial_connection.write(bytes(read_0, 'utf-8'))
            temp0 = serial_connection.read(4)
            temp0 = struct.unpack('<f', temp0)[0]
            temperaturas0[i] = '{}'.format(temp0)

            serial_connection.write(bytes(read_1, 'utf-8'))
            temp1 = serial_connection.read(4)
            temp1 = struct.unpack('<f', temp1)[0]
            temperaturas1[i] = '{}'.format(temp1)

        if flag_header:
            hora_fin = time.time()
            tiempo_mediciones = hora_fin - hora_inicio
            header = {'tiempo_total': measurement_time, 'tiempo_espera': measurement_wait, 'muestras': samples,
                      'tiempo_mediciones': tiempo_mediciones}
            header = json.dumps(header)
            with open(measurement_path_0, 'w') as f:
                f.write(header + '\n')
            with open(measurement_path_1, 'w') as f:
                f.write(header + '\n')
            flag_header = False

        with open(measurement_path_0, 'a') as f:
            hora_ = time.strftime("%H:%M:%S")
            dia_ = time.strftime("%Y/%m/%d")
            f.writelines(",".join(temperaturas0) + ',' + str(hora_) + ',' + str(dia_) + '\n')
        print("Sensor 0: ", temperaturas0)

        with open(measurement_path_1, 'a') as f:
            hora_ = time.strftime("%H:%M:%S")
            dia_ = time.strftime("%Y/%m/%d")
            f.writelines(",".join(temperaturas1) + ',' + str(hora_) + ',' + str(dia_) + '\n')
        print("Sensor 1: ", temperaturas1)

        hora_fin = time.time()
        tiempo_mediciones = hora_fin - hora_inicio
        # print("Tiempo de mediciones: {}".format(tiempo_mediciones))
        espera = measurement_wait - tiempo_mediciones
        time.sleep(espera)


if __name__ == '__main__':
    main()
