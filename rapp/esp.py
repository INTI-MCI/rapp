import serial


class ESP:
    def __init__(self, dev="/dev/ttyUSB0", b=19200, axis=1, reset=True, initpos=0.0, useaxis=[]):
        self.dev = serial.Serial(dev, b)
        self.inuse = useaxis
        if (len(self.inuse) == 0):
            self.inuse = [axis]
        self.defaxis = axis
        if (reset):
            for n in self.inuse:
                self.reset(n)
                print("check error")
                r = self.check_errors()
                if (r != 0):
                    print("Error while setting up controller, error # %d" % r)
                if (initpos != 0):
                    self.setpos(initpos)
                    r = self.check_errors()
                    if (r != 0):
                        print("Error while setting up controller, error # %d" % r)

    def reset(self, axis):
        self.dev.write("{0}OR;{0}WS0\r".format(axis).encode())

    def check_errors(self):
        self.dev.write("TE?\r".encode())
        print("comando enviado")
        output = float(self.dev.readline())
        print("respuesta recibida")
        return output

    def getpos(self, axis=None):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis
        self.dev.write("{0}TP\r".format(a).encode())
        return float(self.dev.readline())

    def setpos(self, pos, axis=None):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis
        print("setting to %f" % pos)
        self.dev.write("{0}PA{1:.4f};{0}WS;{0}TP\r".format(a, pos).encode())
        return float(self.dev.readline())

    def position(self, pos=None, axis=None):
        if (isinstance(pos, (float, int))):
            self.setpos(pos, axis)
            self.getpos()
            self.setpos(pos, axis)
        return self.getpos(axis)
