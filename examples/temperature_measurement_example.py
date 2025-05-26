import serial
import time
import struct

"""
This example provides a simple test for temperature measurements with DS18B20 sensor
"""

serial_port = 'COM3'
baudrate = 57600
timeout = 1

CMD_TEMPLATE = "{measurement};{ch0};{ch1};{samples}\n"

adc = serial.Serial(serial_port, baudrate=baudrate, timeout=timeout)

print(adc.name)
# print(adc.baudrate)
# print(adc.is_open)
adc.reset_input_buffer()

while True:
    adc.write(b'1\n')
    leido = adc.readline()
    if leido != b'':
        leido2 = adc.readline()
        print(leido2)
        leido3 = adc.readline()
        print(leido3)
        break
print('Puerto abierto')
cmd_req_temp = 'req-temp;0;\n'
cmd_temp = 'temp;0;\n'
cmd_complete = 'complete?\n'
# cmd_adc = CMD_TEMPLATE.format(measurement='adc?', ch0=1, ch1=1, samples=3)

tiempos_mediciones = []

for j in range(10):
    tiempos_totales = []
    for i in range(10):
        adc.reset_input_buffer()
        ask = True

        print('------------------')
        print('Iteracion {}'.format(i))

        adc.write(bytes(cmd_req_temp, 'utf-8'))
        tiempo_request = time.time()
        # time.sleep(0.5)

        adc.write(bytes(cmd_temp, 'utf-8'))
        tiempo_get_temp = time.time()
        temp = adc.read(4)
        temperature = struct.unpack('<f', temp)[0]
        print('Temperatura antes del ask', temperature)
        # temp = adc.readline() #readline needs termination character from serial to work properly
        tiempo_read = time.time()

        while ask == True:
            adc.write(bytes(cmd_complete, 'utf-8'))
            ask = adc.read(1)
        tiempo_ask = time.time()

        adc.write(bytes(cmd_temp, 'utf-8'))
        temp = adc.read(4)
        tiempo_total = tiempo_ask - tiempo_request
        print('Tiempo entre request y read: {}'.format(tiempo_read - tiempo_request))
        print('Tiempo entre primer get_temp y read: {}'.format(tiempo_read - tiempo_get_temp))
        print('Tiempo entre request y ask: {}'.format(tiempo_total))

        tiempos_totales.append(tiempo_total)

        temperature = struct.unpack('<f', temp)[0]
        print('Temperatura después del ask', temperature)
        # time.sleep(0.5)
    tiempo_promedio = sum(tiempos_totales) / len(tiempos_totales)
    tiempos_mediciones.append(tiempo_promedio)

print(tiempos_mediciones)
print('Tiempo promedio de mediciones: {}'.format(sum(tiempos_mediciones) / len(tiempos_mediciones)))

tiempos_sensor_placa_indice = [1.045606756210327, 1.0434666872024536, 1.0459917783737183, 1.0466989040374757, 1.0443415880203246, 1.0449856519699097, 1.0468235731124877, 1.0483842849731446, 1.0479555130004883, 1.0446739673614502]
promedio_tiempos_sensor_placa_indice = sum(tiempos_sensor_placa_indice) / len(tiempos_sensor_placa_indice)
print('Tiempo promedio sensor placa indice: {}'.format(promedio_tiempos_sensor_placa_indice))

tiempos_sensor_vaina_direccion = [1.0285144567489624, 1.0274401903152466, 1.0255480766296388, 1.0290863990783692, 1.031200122833252, 1.0302689790725708, 1.0280239582061768, 1.0299288034439087, 1.0305347681045531, 1.0302460193634033]
promedio_tiempos_sensor_vaina_direccion = sum(tiempos_sensor_vaina_direccion) / len(tiempos_sensor_vaina_direccion)
print('Tiempo promedio sensor vaina direccion: {}'.format(promedio_tiempos_sensor_vaina_direccion))

adc.close()
