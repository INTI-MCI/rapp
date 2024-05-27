import pyvisa
from ThorlabsPM100 import ThorlabsPM100
import time
import logging

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

        # Configure for voltage measurement
        #self.configure_voltage = self._pd.configure.scalar.voltage.dc()
        #self.voltage_range = self._pd.sense.voltage.dc.range.upper
        #self.auto_voltage_range = self._pd.sense.voltage.dc.range.auto

        self.configure_voltage = self._pd.configure.scalar.power()
        self._pd.sense.power.dc.range.auto = 'ON'
        self.voltage_range = self._pd.sense.power.dc.range.upper

    @classmethod
    def build(cls, resource):
        rm = pyvisa.ResourceManager()
        if resource in rm.list_resources():
            return cls(resource, rm)
        else:
            return None

    def list_resources():
        rm = pyvisa.ResourceManager()
        print(rm.list_resources())
        rm.close()

    def get_voltage(self):
        if self._pd.getconfigure == 'VOLT':
            return self._pd.read
        else:
            raise PM100Error("PM100 not configured to measure voltage.")

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

    def try_delayed_measurement(self, average_count, wait_time=1e-3, timeout=1):
        old_measurement = self._pd.read
        print(f"Old measurement using read: {old_measurement}")
        print("Starting new read.")
        self.set_average_count(average_count)

        # volt_m = self._pd.measure.scalar.power()
        volt_m = self._pd.measure.scalar.voltage.dc()
        print(f"Old measurement using measure: {volt_m}")
        time_ini = time.time()

        """First try using OPC. But fetch waits for the measurement..."""
        """
        self._pd._write('*OPC')
        while True:
            if time.time() - time_ini >= timeout:
                print(f"{__name__} : Timeout ({timeout} s)")
                break
            opc_bit = self._pd._ask('*OPC?')
            if opc_bit:
                # Fetch waits until a new measurement is available
                new_measurement = self._pd.fetch
                time_end = time.time()
                print(f"OPC was set in {time_end - time_ini:.2f} s. New measurement: "
                      f"{new_measurement}")
                break
            time.sleep(wait_time)
        """

        self._pd.initiate.immediate()
        time.sleep(wait_time)
        new_measurement = self._pd.fetch
        time_end = time.time()
        print(f"Fetch completed after {time_end - time_ini:.2f} s. New measurement: "
              f"{new_measurement}")
