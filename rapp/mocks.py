import random


class SerialMock:
    """Mock object for serial.Serial."""
    def readline(self):
        return '{}'.format(random.randint(0, 30e3)).encode('utf-8')

    def write(self, v):
        pass

    def read(self, n_bytes):
        return bytes(range(n_bytes))

    def close(self):
        pass

    def flushInput(self):
        pass


class ADCMock:
    """Mock object for serial.Serial."""
    serial_mock = SerialMock()

    def acquire(self, samples, **kwargs):
        ch0 = [int(self.serial_mock.readline()) * 0.125/1000 for _ in range(samples)]
        ch1 = [int(self.serial_mock.readline()) * 0.125/1000 for _ in range(samples)]

        return {'C0': ch0, 'C1': ch1}

    def close(self):
        pass

    def flush_input(self):
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

    def sethomevel(self, vel, axis=None):
        pass
