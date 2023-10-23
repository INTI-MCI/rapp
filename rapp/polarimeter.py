import serial
import sys
import os


MEASUREMENT_MODE = 'new_file'
path = 'test.txt'

FILE_NAME = 'test.txt'
FILE_HEADER = "ANALYZER ANGLE [°], A0 [V], A1 [V]"
FILE_ROW = "{},{},{}"

DATA_POINTS_PER_ANGLE = 10
ANGLES_TO_MEASURE = [0]

SERIAL_DEVICE = 'COM3'
SERIAL_BAUDRATE = 57600
SERIAL_TIMEOUT = 0.1


def main(overwrite='n'):
    serialport = serial.Serial(SERIAL_DEVICE, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)

    if os.path.exists(FILE_NAME):
        if overwrite == 'y':
            file = open(FILE_NAME, 'w')
            file.write(FILE_HEADER)
            file.write('\n')
        else:
            print("El archivo ya existe y se optó por no sobreescribir.")
            exit(0)
    else:
        file = open(FILE_NAME, 'a')

    for angle in ANGLES_TO_MEASURE:
        # TODO: move analyzer to the angle.
        i = 0
        while i < DATA_POINTS_PER_ANGLE:
            data_raw = serialport.readline().decode().strip()
            # We need the following IF condition because some values are empty.
            # TODO: Check sleep statements in arduino code...those can be causing this.
            if data_raw:
                a0, a1 = data_raw.split(",")
                row = FILE_ROW.format(angle, a0, a1)
                print(row)
                file.write(row)
                file.write('\n')
                i = i + 1

    file.close()
    serialport.close()


if __name__ == '__main__':
    overwrite = sys.argv[1]
    main(overwrite=overwrite)