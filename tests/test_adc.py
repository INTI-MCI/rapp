from rapp import adc
from rapp import mocks

import pytest  # noqa
import serial


def test_adc_init():
    serial_mock = mocks.SerialMock()
    adc.ADC(serial_mock)


def test_build(monkeypatch):
    with pytest.raises(adc.ADCError):
        adc.ADC.build(dev='BADDEVICE')

    monkeypatch.setattr(serial, "Serial", lambda *x, **y: mocks.SerialMock())
    adc.ADC.build()


def test_acquire():
    with pytest.raises(adc.ADCError):
        ad = adc.ADC(mocks.SerialMock(), ch0=False, ch1=False)

    ad = adc.ADC(mocks.SerialMock(), ch0=True, ch1=True)

    with pytest.raises(adc.ADCError):
        ad.acquire(0)

    SAMPLES = 10
    data = ad.acquire(SAMPLES, flush=False)
    data = ad.acquire(SAMPLES)

    assert len(data) == SAMPLES

    for channel in zip(*data):
        for value in channel:
            print(value)
            assert isinstance(value, float)
            assert 0 <= value <= ad.max_V

    ad = adc.ADC(mocks.SerialMock(), in_bytes=False)
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
