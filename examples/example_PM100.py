# import logging

# import usbtmc
import pyvisa
from ThorlabsPM100 import ThorlabsPM100

""" .rules file in /etc/udev/rules.d

## USB Test Measurement Class (USBTMC) instruments ##

KERNEL=="usbtmc/*", MODE="0666", GROUP="dialout"
KERNEL=="usbtmc[0-9]*", MODE="0666", GROUP="dialout"

# ThorLabs PM100D laser power meter
SUBSYSTEM=="usb", ATTR{idVendor}=="1313", ATTR{idProduct}=="8079", GROUP="dialout", MODE="0666"

*********************************************************
Then, in terminal:
sudo usermod -a -G dialout username

Reboot
"""


# logging.basicConfig(level=logging.DEBUG)
# Get resource string
# rm = pyvisa.ResourceManager("@py")
# print(rm.list_resources())
# print()

PM100_DEVICE = "/dev/usbtmc1"
PM100_RESOURCE_STRING = "USB0::4883::32889::P1000529::0::INSTR"

""" Test using pyserial. USB TMC devices don't work with typical serial connection."""

# serial_connection = serial.Serial(port=PM100_PORT, baudrate=PM100_BAUDRATE, timeout=3)

# serial_connection.write("*IDN?\n".encode())
# errmsg = serial_connection.readline()
# print(errmsg)

# serial_connection.close()

""" Trying simple use of usbtmc device. Not getting anything back."""
# usbtmc = open(PM100_DEVICE, mode="r+")
# print(usbtmc)
# usbtmc.write("*IDN?")
# id_read = usbtmc.readline().strip()

# print(f"ID: {id_read}")
# usbtmc.close()

""" Trying with usbtmc library. """
# instr = usbtmc.Instrument(1313, 8079)  # Device not found
# instr = usbtmc.Instrument(PM100_RESOURCE_STRING) # Invalid resource string
# print(instr.ask("*IDN?"))
# instr.close()

""" Trying with PM100 library

# Not using
from ThorlabsPM100 import ThorlabsPM100, USBTMC
inst = USBTMC(device="/dev/usbtmc0")
power_meter = ThorlabsPM100(inst=inst)
"""
rm = pyvisa.ResourceManager()
print("All resources:", rm.list_resources(), "\n")

inst = rm.open_resource(
    PM100_RESOURCE_STRING,
    timeout=10000,
    write_termination="\n",
    read_termination="\n",
)  # timeout in ms
print("ID: ", inst.query("*IDN?"))
power_meter = ThorlabsPM100(inst=inst)

print("Measurement type :", power_meter.getconfigure)
print("Current value    :", power_meter.read)
print(power_meter.sense.average.count)  # read property
power_meter.sense.average.count = 10  # write property
power_meter.system.beeper.immediate()  # method
