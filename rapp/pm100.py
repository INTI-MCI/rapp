import logging
import time

import pyvisa
from ThorlabsPM100 import ThorlabsPM100

from rapp.mocks import ThorlabsPM100Mock

logger = logging.getLogger(__name__)


class PM100Error(Exception):
    pass


class PM100:
    """Encapsulates the communication with the Thorlabs PM100 detector.

    Args:
        detector: Thorlabs PM100 detector.
    """
    TIME_PER_SAMPLE_MS = 3

    def __init__(self, detector: ThorlabsPM100):
        self._detector = detector
        self._detector.sense.power.dc.range.auto = 'ON'

        self.line_frequency = self._detector.system.lfrequency
        self.sensor_id = self._detector.system.sensor.idn
        self.low_pass_filter_state = self._detector.input.pdiode.filter.lpass.state
        self.configure_power = self._detector.configure.scalar.power()
        self.power_range = self._detector.sense.power.dc.range.upper

    @classmethod
    def average_count_from_duration(cls, duration):
        return int(duration / cls.TIME_PER_SAMPLE_MS * 1e3)

    @classmethod
    def build(
        cls, resource, wavelength=633, avg_count=1, duration=None, timeout=25000, rm=None,
        mock=False, **kwargs
    ):
        """Builds an PM100 object.
        Args:
            resource: name of the VISA resource.
            wavelength: operation wavelength [nm].
            avg_count: amount of averaged samples per measurement. Each sample takes 3 ms.
            duration: the time (in seconds) the measurement should last.
                If provided, avg_count parameter is ignored and is calculated from this one.
            timeout: [ms].
            rm: visa resource manager to use.
            mock: if true, mocks the ThorlabsPM100 object.
        """
        logger.warning("Using PM100 mock object.")

        if mock:
            detector = ThorlabsPM100Mock()
        else:
            if rm is None:
                rm = pyvisa.ResourceManager()

            if resource not in rm.list_resources():
                raise PM100Error("VISA Resource {} not found.".format(resource))

            inst = rm.open_resource(
                resource,
                timeout=timeout,
                # write_termination="\n",
                read_termination="\n",
            )

            detector = ThorlabsPM100(inst=inst)

        if duration is not None:
            avg_count = cls.average_count_from_duration(duration)
            detector.sense.average.count = avg_count

        detector.sense.correction.wavelength = wavelength

        return cls(detector, **kwargs)

    def get_power(self):
        if not self._detector.getconfigure == 'POW':
            raise PM100Error("PM100 not configured to measure power.")

        return self._detector.read

    def start_measurement(self, duration=None):
        """Starts a measurement.

        Args:
            duration: time (in seconds) the measurement should last.
        """
        if duration is not None:
            self._detector.sense.average.count = self.average_count_from_duration(duration)

        self._detector.initiate.immediate()

    def fetch_measurement(self, tolerable_fetch_time=0.01):
        """Fetches the measurement made.

        Args:
            tolerable_fetch_time: max time a fetch should take. If surpassed, a warning is logged.
        """
        start_time = time.time()
        data = self._detector.fetch
        elapsed_time = time.time() - start_time

        if elapsed_time > tolerable_fetch_time:
            logger.warning(f"Thorlabs PM100 fetch time was {elapsed_time:.2f} s.")

        return data

    def close(self):
        # Bad: should be using a method of ThorlabsPM100 object to close the connection.
        # TODO: find out how to do that.
        pyvisa.ResourceManager().close()
