import serial


arduino = serial.Serial('/dev/ttyACM0', 57600, timeout=0.1)

while True:
    data_raw = arduino.readline().decode().strip()
    if data_raw:
        print(data_raw)

arduino.close()
