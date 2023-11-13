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

    def setvel(self, vel=2, axis=None):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis

        print("setting velocity to %f" % vel)

        self.dev.write("{0}VA{1:.4f};{0}TV\r".format(a, vel).encode())
        return float(self.dev.readline())

    def sethomevel(self, vel=2, axis=None):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis

        print("setting home velocity to %f" % vel)

        self.dev.write("{0}OL{1}\r".format(a, vel).encode())
        self.dev.write("{0}OH{1}; {0}OH?\r".format(a, vel).encode())

        return float(self.dev.readline())

    def setacc(self, acc=2, axis=None):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis

        print("setting acceleration to %f" % acc)

        self.dev.write("{0}CA{1:.4f};\r".format(a, acc).encode())

    def check_errors(self):
        self.dev.write("TE?\r".encode())
        output = float(self.dev.readline())
        return output

    def getpos(self, axis=None):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis
        self.dev.write("{0}TP\r".format(a).encode())
        return float(self.dev.readline())

    def setpos(self, pos, ws=0, axis=None, relative=True):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis
        print("setting to %f" % pos)

        command = "PA"
        if relative:
            command = "PR"

        commands = "{0}{c}{1:.4f};{0}WS{ws};{0}WP{1:.4f};{0}TP\r".format(a, pos, c=command, ws=ws)
        print(commands)

        self.dev.write(commands.encode())
        return float(self.dev.readline())

    def position(self, pos=None, axis=None):
        if (isinstance(pos, (float, int))):
            self.setpos(pos, axis)
            self.getpos()
            self.setpos(pos, axis)
        return self.getpos(axis)
