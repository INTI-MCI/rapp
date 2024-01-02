import time
import random


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
