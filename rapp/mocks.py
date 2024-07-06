import time
import random


class SerialMock:
    """Mock object for serial.Serial."""

    def __init__(self, delay=0):
        self.delay = delay

    def readline(self):
        return '{}'.format(0).encode('utf-8')

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
    def __init__(self, delay=0):
        self.system = SystemPM100Mock()
        self.sense = SensePM100Mock()
        self.input = InputPM100Mock()
        self.configure = ConfigurePM100Mock()
        self.initiate = InitiatePM100Mock()
        self.getconfigure = 'POW'
        # self._read = 0
        # self._fetch = 0

        self._delay = delay

    @property
    def read(self):
        return random.gammavariate(alpha=1, beta=1)

    @property
    def fetch(self):
        time.sleep(self._delay)
        return random.gammavariate(alpha=1, beta=1)

    def close(self):
        pass


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
    def immediate(self):
        pass
