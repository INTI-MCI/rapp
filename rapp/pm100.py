import pyvisa
from ThorlabsPM100 import ThorlabsPM100


class PM100Error(Exception):
    pass


class PM100:
    def __init__(self, resource, rm, average_count=1, wavelength=633):
        """
        resource: if unknown, use list_resources
        rm: VISA Resource Manager
        average_count: amount of averaged samples. Each of them taked 3 ms
        wavelength: operation wavelength [nm]
        """
        self.resource = resource
        self._rm = rm

        self.inst = self._rm.open_resource(
            self.resource,
            timeout=1000,
            write_termination="\n",
            read_termination="\n",
        )

        self._pd = ThorlabsPM100(inst=self.inst)

        self.line_frequency = self._pd.system.lfrequency
        self.sensor_id = self._pd.system.sensor.idn
        self.average_count = average_count
        self._pd.sense.average.count = average_count
        self.wavelength = wavelength
        self._pd.sense.correction.wavelength = wavelength
        self.bandwidth = self._pd.input.pdiode.filter.lpass.state

        # Configure for voltage measurement
        self.configure_voltage = self._pd.configure.scalar.voltage.dc()
        self.voltage_range = self._pd.sense.voltage.dc.range.upper
        self.auto_voltage_range = self._pd.sense.voltage.dc.range.auto

    @classmethod
    def build(cls, resource):
        rm = pyvisa.ResourceManager()
        if resource in rm.list_resources():
            return cls(resource, rm)
        else:
            return None

    @classmethod
    def list_resources(cls):
        rm = pyvisa.ResourceManager()
        print(rm.list_resources())
        rm.close()

    def get_voltage(self):
        if self._pd.getconfigure == 'VOLT':
            return self._pd.read
        else:
            raise PM100Error("PM100 not configured to measure voltage.")
