class SerialMock:
    """Mock object for serial.Serial."""
    def readline(self):
        return b'1340'

    def write(self, v):
        pass

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
