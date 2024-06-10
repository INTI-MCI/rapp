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
        self.system = SystemPM100Mock()
        self.sense = SensePM100Mock()
        self.input = InputPM100Mock()
        self.configure = ConfigurePM100Mock()
        self.getconfigure = 'POW'
        self._read = 0
        self.initiate = InitiatePM100Mock()
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


class SystemPM100Mock:
    def __init__(self) -> None:
        self.lfrequency = 50
        self.sensor = SystemSensorPM100Mock()


class SystemSensorPM100Mock:
    def __init__(self) -> None:
        self.idn = "Mocked sensor"


class SensePM100Mock:
    def __init__(self) -> None:
        self.average = SenseAveragePM100Mock()
        self.correction = SenseCorrectionPM100Mock()
        self.power = SensePowerPM100Mock()


class SenseAveragePM100Mock:
    def __init__(self) -> None:
        self.count = 1


class SenseCorrectionPM100Mock:
    def __init__(self) -> None:
        self.wavelength = 633


class SensePowerPM100Mock:
    def __init__(self) -> None:
        self.dc = SensePowerDCPM100Mock()


class SensePowerDCPM100Mock:
    def __init__(self) -> None:
        self.range = SensePowerDcRangePM100Mock()


class SensePowerDcRangePM100Mock:
    def __init__(self) -> None:
        self.auto = 'ON'
        self.upper = 1


class InputPM100Mock:
    def __init__(self) -> None:
        self.pdiode = InputPdiodePM100Mock()


class InputPdiodePM100Mock:
    def __init__(self) -> None:
        self.filter = InputPdiodeFilterPM100Mock()


class InputPdiodeFilterPM100Mock:
    def __init__(self) -> None:
        self.lpass = InputPdiodeFilterLpassPM100Mock()


class InputPdiodeFilterLpassPM100Mock:
    def __init__(self) -> None:
        self.state = 1


class ConfigurePM100Mock:
    def __init__(self) -> None:
        self.scalar = ConfigureScalarPM100Mock()


class ConfigureScalarPM100Mock:
    def power(self):
        return random.gammavariate(alpha=1, beta=1)


class InitiatePM100Mock:
    def __init__(self):
        self.start_time = time.time()

    def immediate(self):
        self.start_time = time.time()
