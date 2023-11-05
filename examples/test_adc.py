import time
import serial

# ADC_DEVICE = 'COM4'
ADC_DEVICE = '/dev/ttyACM0'

ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1

N_SAMPLES = 8


adc = serial.Serial(ADC_DEVICE, ADC_BAUDRATE, timeout=ADC_TIMEOUT)

print("Esperando segundos...")
# Arduino resets when a new serial connection is made.
# We need to wait, otherwise we don't recieve anything.
# Less than 2 seconds does not work.
# TODO: check if we can avoid that arduino resets.
time.sleep(2)

adc.write(bytes(str(N_SAMPLES), 'utf-8'))
# Sending directly the numerical value didn't work.
# See: https://stackoverflow.com/questions/69317581/sending-serial-data-to-arduino-works-in-serial-monitor-but-not-in-python  # noqa

print("Adquiriendo...")
data = []
while len(data) < N_SAMPLES:
    try:
        value = adc.readline().decode().strip()
        if value:
            print("A0: {}".format(value))
            data.append(value)
    except (ValueError, UnicodeDecodeError) as e:
        print(e)

adc.close()
