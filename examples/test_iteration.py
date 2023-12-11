import sys
import time
import serial

from rapp.utils import progressbar


def bits_to_volts(value):
    return value * 0.125 / 1000


def test_append(size,  in_bytes=True):
    data = []
    b = 57600
    timeout = 0.1
    dev = 'COM4'

    adc = serial.Serial(dev, baudrate=b, timeout=timeout)

    adc.flushInput()

    print("Waiting 2 seconds to start...")
    time.sleep(2)

    print("Measuring {} samples...".format(size))
    adc.write(bytes("{}s".format(size), 'utf-8'))
    for i in progressbar(range(size), size=100):
        if in_bytes:
            value = adc.read(2)
            value = int.from_bytes(value, byteorder='big', signed=True)
        else:
            value = adc.readline().decode().strip()
            value = int(value)

        data.append(bits_to_volts(value))


def main():
    test_append(int(sys.argv[1]))


if __name__ == '__main__':
    main()
