import time
import random


class SerialMock:
    """Mock object for serial.Serial."""

    def __init__(self, delay=0):
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

    def open(self):
        pass

    def flushInput(self):
        pass

    def _random_value(self):
        return random.randint(1000, 5000)


class ThorlabsPM100Mock:
    def __init__(self):
        self._innerA = innerAThorlabs()
        self.system = self._innerA
        self.sense = self._innerA
        self.input = self._innerA
        self.configure = self._innerA
        self.getconfigure = 'POW'
        self._read = 0
        self.initiate = self._innerA
        self._fetch = 0

    @property
    def read(self):
        time.sleep(self.get_delay_time())
        return random.gammavariate(alpha=1, beta=1)

    @property
    def fetch(self):
        wait_time = self.get_delay_time() - (time.time() - self.initiate.start_time)
        if wait_time > 0:
            time.sleep(wait_time)
        self._fetch = random.gammavariate(alpha=1, beta=1)
        return self._fetch

    def get_delay_time(self):
        return 3e-3 * self.sense.average.count


class innerAThorlabs:
    def __init__(self):
        self.innerB = innerBThorlabs()
        self.start_time = time.time()
        # System
        self.lfrequency = 50
        self.sensor = self.innerB
        # Sense
        self.average = self.innerB
        self.correction = self.innerB
        self.power = self.innerB
        # Input
        self.pdiode = self.innerB
        # Configure
        self.scalar = self.innerB

    def immediate(self):
        self.start_time = time.time()


class innerBThorlabs:
    def __init__(self) -> None:
        self.innerC = innerCThorlabs()
        # System
        self.idn = "Mocked sensor"
        # Sense
        self.count = 1
        self.wavelength = 633
        self.dc = self.innerC
        # Input
        self.filter = self.innerC

    def power(self):
        # Configure
        return random.gammavariate(alpha=1, beta=1)


class innerCThorlabs:
    def __init__(self) -> None:
        self.innerD = innerDThorlabs()
        # Input
        self.lpass = self.innerD
        # Sense
        self.range = self.innerD


class innerDThorlabs:
    def __init__(self) -> None:
        self.state = 1
        self.auto = 'ON'
        self.upper = 1
