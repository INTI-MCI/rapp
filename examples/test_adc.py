import time
import serial

from rapp.utils import timing


# ADC_DEVICE = 'COM4'
ADC_DEVICE = '/dev/ttyACM0'

ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_MULTIPLIER_mV = 0.125

WAIT_TIME_AFTER_CONNECTION = 2


def bits_to_volts(value):
    return value * ADC_MULTIPLIER_mV / 1000


@timing
def acquire(adc, n_samples):
    adc.write(bytes(str(n_samples), 'utf-8'))
    # Sending directly the numerical value didn't work.
    # See: https://stackoverflow.com/questions/69317581/sending-serial-data-to-arduino-works-in-serial-monitor-but-not-in-python  # noqa

    data = []
    while len(data) < n_samples:
        try:
            value = int(adc.readline().decode().strip())
            if value:
                value = bits_to_volts(value)
                print("A0: {}".format(value))
                data.append(value)
        except (ValueError, UnicodeDecodeError) as e:
            print(e)


def main():
    adc = serial.Serial(ADC_DEVICE, ADC_BAUDRATE, timeout=ADC_TIMEOUT)
    adc.flushInput()

    print("Waiting {} seconds...".format(WAIT_TIME_AFTER_CONNECTION))
    # Arduino resets when a new serial connection is made.
    # We need to wait, otherwise we don't recieve anything.
    # Less than 2 seconds does not work.
    # TODO: check if we can avoid that arduino resets.
    time.sleep(WAIT_TIME_AFTER_CONNECTION)

    print("Adquiriendo...")
    acquire(adc, n_samples=855)

    adc.close()


if __name__ == '__main__':
    main()
