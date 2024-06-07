import pyvisa
from ThorlabsPM100 import ThorlabsPM100
import logging
from rapp.mocks import ThorlabsPM100Mock

logger = logging.getLogger(__name__)


class PM100Error(Exception):
    pass


class PM100:
    def __init__(self, resource, rm, average_count=1, wavelength=633, timeout=25000):
        """
        resource: if unknown, use list_resources
        rm: VISA Resource Manager
        average_count: amount of averaged samples. Each of them taked 3 ms
        wavelength: operation wavelength [nm]
        timeout: [ms]
        """
        self.resource = resource
        self._rm = rm

        if resource == "mock":
            self._pd = ThorlabsPM100Mock()
        else:
            self.inst = self._rm.open_resource(
                self.resource,
                timeout=timeout,
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
        self.low_pass_filter_state = self._pd.input.pdiode.filter.lpass.state

        self.configure_power = self._pd.configure.scalar.power()
        self._pd.sense.power.dc.range.auto = 'ON'
        self.power_range = self._pd.sense.power.dc.range.upper

    @classmethod
    def build(cls, resource):
        rm = pyvisa.ResourceManager()
        if resource in (*(rm.list_resources()), "mock"):
            return cls(resource, rm)
        else:
            return None

    def list_resources():
        rm = pyvisa.ResourceManager()
        print(rm.list_resources())
        rm.close()

    def get_power(self):
        if self._pd.getconfigure == 'POW':
            return self._pd.read
        else:
            raise PM100Error("PM100 not configured to measure power.")

    def close(self):
        self._rm.close()

    def set_average_count(self, average_count):
        if self.average_count != average_count:
            self.average_count = average_count
            self._pd.sense.average.count = average_count

    def start_measurement(self):
        self._pd.initiate.immediate()

    def fetch_measurement(self):
        return self._pd.fetch
