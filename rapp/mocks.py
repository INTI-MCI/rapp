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
