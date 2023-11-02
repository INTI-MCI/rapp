import os
import sys
import time

from datetime import datetime

from esp import ESP  # noqa
import serial        # noqa


FILE_HEADER = "ANGLE [°], A0 [V], A1 [V], DATETIME"
FILE_ROW = "{angle} {a0} {a1} {datetime}"

OUTPUT_DIR = "mediciones"

ADC_DEVICE = 'COM4'
ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_MAX_VAL = 4.096


class SerialMock:
    """Mock object for serial.Serial."""
    def readline(self):
        return b'3.4567,2.3422'

    def close(self):
        pass

    def flushInput(self):
        pass


class ESPMock:
    """Mock object for esp.ESP."""
    dev = SerialMock()

    pos = 0
    vel = 2

    def setpos(self, pos, axis=None):
        self.pos = pos
        return pos

    def getpos(self, axis=None):
        return self.pos

    def setvel(self, vel, axis=None):
        self.vel = vel
        return vel


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


def parse_data(data):
    if not data:
        raise ValueError("Data is empty!")

    data_list = data.split(",")
    if len(data_list) != 2:
        raise ValueError("Expected 2 values! Got {}.".format(len(data_list)))

    a0, a1 = data_list

    if (
        not (-ADC_MAX_VAL <= float(a0) <= ADC_MAX_VAL) or
        not (-ADC_MAX_VAL <= float(a1) <= ADC_MAX_VAL)
    ):
        raise ValueError('Values out of the range [-{}, {}]'.format(ADC_MAX_VAL, ADC_MAX_VAL))

    return a0, a1


def main(n_cycles=1, step=10, n_points=10, delay=0, filename='test', verbose=False):

    angles = [i for i in range(0, 360 * n_cycles, step)]
    filename = "{}-n_cycles{}-step{}-n_points{}.txt".format(filename, n_cycles, step, n_points)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, filename)

    # Mock objects to test when we don't have the device connected.
    # TODO: create a unit test and use them from there.
    serialport = SerialMock()
    analyzer = ESPMock()

    # serialport = serial.Serial(ADC_DEVICE, ADC_BAUDRATE, timeout=ADC_TIMEOUT)
    # analyzer = ESP(dev="COM3", b=921600, axis=1, reset=True)

    time.sleep(2)           # Why this?
    analyzer.setvel(vel=4)

    overwrite = False
    if os.path.exists(filename):
        i = input(
            "El archivo ya existe. Querés borrar el archivo? s/n (ENTER es NO): ")
        if i == 's':
            overwrite = True

    file = create_or_open_file(filename, overwrite)

    for angle in angles:
        analyzer.setpos(angle)

        time.sleep(1)  # wait for position to stabilize
        # TODO: assert that the new position is the expected, up to N decimal positions?
        print(f"Measuring at position: {analyzer.getpos()}")

        serialport.flushInput()  # Clear buffer. Otherwise messes up measurements at the beginning.

        i = 0
        while i < n_points:
            data = serialport.readline().decode().strip()

            try:
                a0, a1 = parse_data(data)
            except ValueError as e:
                print("Found error in data: {}. data: {}. Skipping...".format(e, data))
                continue

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
    n_cycles = int(sys.argv[1])          # how many cycles.
    step = int(sys.argv[2])              # every how many degrees to take a measurement.
    n_points = int(sys.argv[3])          # how many points per angle.
    delay = float(sys.argv[4])           # the delay (in seconds) between each measurement.
    filename = str(sys.argv[5])          # the filename.
    verbose = bool(int(sys.argv[6]))     # whether to print each measurement. 1 (true) - 0 (false).

    # example:
    # python polarimeter.py 1 180 10 0 test.txt 0

    main(n_cycles, step, n_points, delay, filename, verbose)
