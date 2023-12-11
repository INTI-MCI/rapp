import time
import random

from rapp.adc import ADC


class SerialMock:
    """Mock object for serial.Serial."""
    def __init__(self, delay=0.0018):
        self.delay = delay

    def readline(self):
        return '{}'.format(self._random_value()).encode('utf-8')

    def write(self, v):
        pass

    def read(self, n_bytes):
        time.sleep(self.delay)
        value = self._random_value()
        return int(value).to_bytes(length=2, byteorder='big')

    def close(self):
        pass

    def flushInput(self):
        pass

    def _random_value(self):
        return random.randint(1000, 5000)


class ADCMock(ADC):
    """Mock object for serial.Serial."""
    def __init__(self, progressbar=True):
        self._serial = SerialMock()
        self._max_V, self._multiplier_mV = (4.096, 0.125)
        self.progressbar = progressbar

    def close(self):
        pass

    def flush_input(self):
        pass


class ESPMock:
    """Mock object for esp.ESP."""
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

    def sethomevel(self, vel, axis=None):
        pass

    def close(self):
        pass
