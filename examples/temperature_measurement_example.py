import serial
import time
import struct

"""
This example provides a simple test for temperature measurements
"""

serial_port = 'COM3'
baudrate = 57600
timeout = 0.5

CMD_TEMPLATE = "{measurement};{ch0};{ch1};{samples}\n"

adc = serial.Serial(serial_port, baudrate=baudrate, timeout=timeout)

print(adc.name)
print(adc.baudrate)
print(adc.is_open)
adc.flushInput()

while True:
    adc.write(b'1\n')
    leido = adc.readline()
    if leido != b'':
        break
print('Puerto abierto')
cmd = 'temp?\n'
# cmd = CMD_TEMPLATE.format(measurement='adc?', ch0=1, ch1=1, samples=3)
print(bytes(cmd, 'utf-8'))
for i in range(5):
    adc.write(bytes(cmd, 'utf-8'))
    # temp = adc.read(4)
    temp = adc.readline()
    print(temp)
    temperature = struct.unpack('<f', temp)[0]
    print(temperature)
    time.sleep(0.01)
adc.close()
