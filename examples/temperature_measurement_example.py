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
adc.flushInput()

while True:
    adc.write(b'1\n')
    leido = adc.readline()
    if leido != b'':
        break
print('Puerto abierto')
cmd1 = 'req-temp?\n'
cmd2 = 'temp?\n'
cmd3 = 'complete?\n'
# cmd = CMD_TEMPLATE.format(measurement='adc?', ch0=1, ch1=1, samples=3)

for i in range(5):
    adc.reset_input_buffer()
    ask = True

    adc.write(bytes(cmd1, 'utf-8'))
    tiempo_request = time.time()
    # time.sleep(0.5)

    adc.write(bytes(cmd2, 'utf-8'))
    tiempo_get_temp = time.time()
    temp = adc.read(4)
    temperature = struct.unpack('<f', temp)[0]
    print('Temperatura antes del ask', temperature)
    # temp = adc.readline() #readline needs termination character from serial to work properly
    tiempo_read = time.time()

    while ask == True:
        adc.write(bytes(cmd3, 'utf-8'))
        ask = adc.read(1)
    tiempo_ask = time.time()

    adc.write(bytes(cmd2, 'utf-8'))
    temp = adc.read(4)
    print('Tiempo request-read: {}'.format(tiempo_read - tiempo_request))
    print('Tiempo 1º get_temp-read: {}'.format(tiempo_read - tiempo_get_temp))
    print('Tiempo request-ask: {}'.format(tiempo_ask - tiempo_request))

    temperature = struct.unpack('<f', temp)[0]
    print('Temperatura después del ask', temperature)
    # time.sleep(0.5)
adc.close()
