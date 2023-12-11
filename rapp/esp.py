import logging
import serial

logger = logging.getLogger(__name__)


class ESP:
    def __init__(self, dev="/dev/ttyUSB0", b=19200, axis=1, reset=False, initpos=0.0, useaxis=[]):
        self.dev = serial.Serial(dev, b)
        self.inuse = useaxis
        if (len(self.inuse) == 0):
            self.inuse = [axis]

        self.defaxis = axis

        logger.debug("Turning motor ON...")
        self.motor_on()

        if reset:
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

    def motor_on(self, axis=None):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis

        self.dev.write("{0}MO; {0}MO?\r".format(a).encode())

        res = float(self.dev.readline())
        return res == 1

    def motor_off(self, axis=False):
        self.dev.write("{0}MF; {0}MF?\r".format(axis).encode())

        res = float(self.dev.readline())
        return res == 1

    def hardware_reset(self):
        self.dev.write("RS\r".encode())

    # units: 0:encoder count, 1:motor step, 7:deg, 9:rad, 10:mrad, 11:murad, ?:report current setts
    def set_units(self, axis, unit):
        self.dev.write("{0}SN{1}\r".format(axis, unit).encode())

    # resolution es la cantidad de decimales en las user units
    def display_res(self, axis, resolution):
        self.dev.write("{0}FP{1}\r".format(axis, resolution).encode())

    def reset(self, axis):
        self.dev.write("{0}OR;{0}WS0\r".format(axis).encode())

    def setvel(self, vel=2, axis=None):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis

        self.dev.write("{0}VA{1:.4f};{0}TV\r".format(a, vel).encode())
        return float(self.dev.readline())

    def sethomevel(self, vel=2, axis=None):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis

        self.dev.write("{0}OL{1}\r".format(a, vel).encode())
        self.dev.write("{0}OH{1}; {0}OH?\r".format(a, vel).encode())

        return float(self.dev.readline())

    def setacc(self, acc=2, axis=None):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis

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

    def setpos(self, pos, ws=0, axis=None, relative=False):
        a = self.defaxis
        if (axis and axis > 0):
            a = axis

        command = "PA"
        if relative:
            command = "PR"

        commands = "{0}{c}{1:.4f};{0}WS{ws};{0}WP{1:.4f};{0}TP\r".format(a, pos, c=command, ws=ws)

        self.dev.write(commands.encode())
        return float(self.dev.readline())

    def position(self, pos=None, axis=None):
        if (isinstance(pos, (float, int))):
            self.setpos(pos, axis)
            self.getpos()
            self.setpos(pos, axis)
        return self.getpos(axis)

    def close(self):
        self.dev.close()
