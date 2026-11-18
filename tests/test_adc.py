import numpy as np
import pytest  # noqa
import serial

from rapp import adc, mocks


def test_adc_init():
    serial_mock = mocks.SerialMock()
    adc.ADC(serial_mock, "ADS_1115")


def test_build(monkeypatch):
    with pytest.raises(adc.ADCError):
        adc.ADC.build(dev='BADDEVICE', adc_used="ADS_1115")

    monkeypatch.setattr(serial, "Serial", lambda *x, **y: mocks.SerialMock())
    adc.ADC.build(adc_used="ADS_1115")


def test_acquire():
    with pytest.raises(adc.ADCError):
        ad = adc.ADC(mocks.SerialMock(), ch0=False, ch1=False, adc_used="ADS_1115")

    ad = adc.ADC(mocks.SerialMock(), ch0=True, ch1=True, adc_used="ADS_1115")

    with pytest.raises(adc.ADCError):
        ad.acquire(0)

    SAMPLES = 10
    data = ad.acquire(SAMPLES, flush=False)
    data = ad.acquire(SAMPLES)

    assert len(data) == SAMPLES

    for channel in zip(*data):
        for value in channel:
            # print(value)
            assert isinstance(value, float)
            assert 0 <= value <= ad.max_V

    ad = adc.ADC(mocks.SerialMock(), in_bytes=False, adc_used="ADS_1115")
    ad.acquire(1)

    ad.close()


def test_short_adc_read_raises_adc_error():
    class EmptyReadSerial(mocks.SerialMock):
        def read(self, n_bytes):
            return b""

    ad = adc.ADC(EmptyReadSerial(), adc_used="ADS_1232")

    with pytest.raises(adc.ADCError, match="Expected 4 bytes, got 0 bytes"):
        ad.acquire(1)


def test_short_self_calibration_response_raises_adc_error():
    class EmptyReadSerial(mocks.SerialMock):
        def read(self, n_bytes):
            return b""

    ad = adc.ADC(EmptyReadSerial(), adc_used="ADS_1232")

    with pytest.raises(adc.ADCError, match="Expected 2 bytes, got 0 bytes"):
        ad.self_calibrate()


def test_active_channels_and_measurement_time():
    ad = adc.ADC(mocks.SerialMock(), ch0=True, ch1=True, adc_used="ADS_1232")

    assert ad.active_channels() == 2
    assert ad.measurement_time(ad.SAMPLE_RATE_24) == pytest.approx(2.0)

    ad = adc.ADC(mocks.SerialMock(), ch0=True, ch1=False, adc_used="ADS_1232")

    assert ad.active_channels() == 1
    assert ad.measurement_time(2 * ad.SAMPLE_RATE_24) == pytest.approx(2.0)


def test_bits_to_volts():
    ad = adc.ADC(mocks.SerialMock(), adc_used="CS_1237", gain=1)

    assert ad._bits_to_volts(None) is None
    assert ad._bits_to_volts(0) == 0
    assert ad._bits_to_volts(2**23) == pytest.approx(ad.max_V)


def test_request_and_read_temperature():
    ad = adc.ADC(mocks.SerialMock(), adc_used="ADS_1232")

    assert ad.request_temperature(0) is True
    assert ad.request_temperature(1, flush=False) is True

    temperature, requested = ad.read_temperature(0)

    assert len(temperature) == 1
    assert isinstance(temperature[0], float)
    assert requested is False


def test_print_read_n_every_dt():
    ad = adc.ADC(mocks.SerialMock(), adc_used="ADS_1232")

    with pytest.raises(adc.ADCError):
        ad.print_read_n_every_dt(channel=0, n_samples=0, dt_mus=100)

    mean_dt = ad.print_read_n_every_dt(channel=0, n_samples=3, dt_mus=100)

    assert np.isfinite(mean_dt)


def test_self_calibrate_reports_adc_errors(caplog):
    class SelfCalibrationErrorSerial(mocks.SerialMock):
        def write(self, value):
            if value == b'self_calibrate;\n':
                self.cmd_received = 'self_calibrate'
                self.buffer = b'\x01\x00\x02\x00'
            else:
                super().write(value)

    ad = adc.ADC(SelfCalibrationErrorSerial(), adc_used="ADS_1232")

    with caplog.at_level("ERROR"):
        ad.self_calibrate()

    assert "ADC0: 1" in caplog.text
    assert "ADC1: 2" in caplog.text


def test_reset_adc():
    class RecordingSerial(mocks.SerialMock):
        def __init__(self):
            super().__init__()
            self.written = []

        def write(self, value):
            self.written.append(value)
            super().write(value)

    serial_mock = RecordingSerial()
    ad = adc.ADC(serial_mock, adc_used="ADS_1232")

    ad.reset_ADC()

    assert serial_mock.written[-1] == b'reset_adc;\n'


def test_recover():
    ad = adc.ADC(mocks.SerialMock(), adc_used="ADS_1232")

    ad.recover()


def test_recover_failure_raises_recovery_error():
    ad = adc.ADC(mocks.SerialMock(), adc_used="ADS_1232")
    ad._serial = object()

    with pytest.raises(adc.ADCRecoveryError, match="Could not recover"):
        ad.recover()


def test_wait_for_connection_timeout():
    with pytest.raises(adc.ADCError, match="Timeout while waiting"):
        adc.ADC(mocks.SerialMock(), adc_used="ADS_1232", timeout_open=0)


def test_wait_for_connection_write_timeout():
    class WriteTimeoutSerial(mocks.SerialMock):
        def write(self, value):
            if value == b'ready?\n':
                raise serial.SerialTimeoutException("timeout")
            super().write(value)

    with pytest.raises(adc.ADCError, match="Timeout while writing"):
        adc.ADC(WriteTimeoutSerial(), adc_used="ADS_1232")


def test_wait_for_connection_answers_not_ready():
    class NotReadySerial(mocks.SerialMock):
        def readline(self):
            return b'no\r\n'

    with pytest.raises(adc.ADCError, match="Timeout while waiting"):
        adc.ADC(NotReadySerial(), adc_used="ADS_1232", timeout_open=0.01)


def test_acquire_returns_none_on_short_channel_data(monkeypatch):
    ad = adc.ADC(mocks.SerialMock(), ch0=True, ch1=False, adc_used="ADS_1232")
    monkeypatch.setattr(ad, "_read_data", lambda n_samples, name='': [])

    assert ad.acquire(2) is None

    ad = adc.ADC(mocks.SerialMock(), ch0=False, ch1=True, adc_used="ADS_1232")
    monkeypatch.setattr(ad, "_read_data", lambda n_samples, name='': [])

    assert ad.acquire(2) is None


def test_read_data_stops_when_measurement_is_none(monkeypatch):
    ad = adc.ADC(mocks.SerialMock(), adc_used="ADS_1232")
    monkeypatch.setattr(ad, "_read_bits", lambda: None)

    assert ad.acquire(2) is None


def test_read_data_ignores_read_errors(monkeypatch):
    ad = adc.ADC(mocks.SerialMock(), adc_used="ADS_1232")
    values = iter([ValueError("bad value"), 1000, 2000, 3000, 4000])

    def read_bits():
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(ad, "_read_bits", read_bits)

    # One of the three requested samples is lost, so the acquisition is incomplete.
    assert ad.acquire(3) is None


def test_read_bytes_serial_exception():
    class BrokenSerial(mocks.SerialMock):
        def read(self, n_bytes):
            raise serial.SerialException("boom")

    ad = adc.ADC(mocks.SerialMock(), adc_used="ADS_1232")
    ad._serial = BrokenSerial()

    with pytest.raises(adc.ADCError, match="Error while reading"):
        ad._read_bytes(4)


class WriteTimeoutSerial(mocks.SerialMock):
    def write(self, value):
        raise serial.SerialTimeoutException("timeout")


def test_acquire_with_one_channel_returns_none_for_the_other(monkeypatch):
    ad = adc.ADC(mocks.SerialMock(), ch0=True, ch1=False, adc_used="ADS_1232")

    data = ad.acquire(2)

    assert len(data) == 2
    assert all(ch1 is None for _, ch1 in data)


def test_print_read_n_every_dt_stops_when_measurement_is_none(monkeypatch):
    ad = adc.ADC(mocks.SerialMock(), adc_used="ADS_1232")
    monkeypatch.setattr(ad, "_read_bits", lambda: None)

    mean_dt = ad.print_read_n_every_dt(channel=0, n_samples=3, dt_mus=100)

    assert np.isfinite(mean_dt)


@pytest.mark.parametrize("method, args", [
    ("acquire", (1,)),
    ("request_temperature", (0,)),
    ("read_temperature", (0,)),
    ("print_read_n_every_dt", (0, 1, 100)),
    ("self_calibrate", ()),
    ("reset_ADC", ()),
])
def test_adc_write_timeouts(method, args):
    ad = adc.ADC(mocks.SerialMock(), adc_used="ADS_1232")
    ad._serial = WriteTimeoutSerial()

    with pytest.raises(adc.ADCError, match="Timeout while writing"):
        getattr(ad, method)(*args)
