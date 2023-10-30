import os
import sys
import time

from datetime import datetime

from esp import ESP  # noqa
import serial        # noqa


FILE_HEADER = "ANGLE [°], A0 [V], A1 [V], DATETIME"
FILE_ROW = "{angle} {a0} {a1} {datetime}"

SERIAL_DEVICE = 'COM3'
SERIAL_BAUDRATE = 57600
SERIAL_TIMEOUT = 0.1


class SerialMock:
    """Mock object for serial.Serial."""
    def readline(self):
        return b'3.4567,2.3422'

    def close(self):
        pass


class ESPMock:
    """Mock object for esp.ESP."""
    dev = SerialMock()

    def setpos(self, pos, axis=None):
        return 0


def create_file(filename):
    file = open(filename, 'w')
    file.write(FILE_HEADER)
    file.write('\n')
    return file


def create_or_open_file(filename, overwrite):
    if not os.path.exists(filename) or overwrite:
        file = create_file(filename)
    else:
        file = open(filename, 'a')

    return file


def main(angles, n_points=10, delay=0, filename='test.txt', verbose=False):

    # Mock objects to test when we don't have the device connected.
    # TODO: create a unit test and use them from there.
    serialport = SerialMock()
    analyzer = ESPMock()

    # serialport = serial.Serial(SERIAL_DEVICE, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
    # analyzer = ESP("COM3", 921600, 1, reset=False)

    overwrite = False
    if os.path.exists(filename):
        i = input(
            "El archivo ya existe. Querés borrar el archivo? s/n (ENTER es NO): ")
        if i == 's':
            overwrite = True

    file = create_or_open_file(filename, overwrite)

    for angle in angles:
        analyzer.setpos(angle)

        i = 0
        while i < n_points:
            data_raw = serialport.readline().decode().strip()

            if data_raw:  # TODO: check this IF is needed now that we removed the arduino delay.
                a0, a1 = data_raw.split(",")
                datetime_ = datetime.now().isoformat()
                row = FILE_ROW.format(angle=angle, a0=a0, a1=a1, datetime=datetime_)

                if verbose:
                    print(row)

                file.write(row)
                file.write('\n')
                i = i + 1

                time.sleep(delay)

    file.close()
    serialport.close()
    analyzer.dev.close()


if __name__ == '__main__':
    n_points = int(sys.argv[1])          # pass how many points to measure.
    delay = float(sys.argv[2])           # pass the delay (in seconds) between measurements.
    filename = str(sys.argv[3])          # pass the filename.
    verbose = bool(int(sys.argv[5]))     # pass 1 (true) or 0 (false).

    # example:
    # python polarimeter.py 10 0 test.txt 0

    # angles = [i % 360 for i in range(0, 360 * 2, 10)]

    angles = [0]
    main(angles, n_points, delay, filename, verbose)
