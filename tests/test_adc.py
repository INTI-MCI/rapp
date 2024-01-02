from rapp import adc
from rapp import mocks

import pytest  # noqa
import serial


def test_adc_init():

    with pytest.raises(adc.ADCError):
        adc.ADC(wait=0)

    adc.ADC(mocks.SerialMock(), wait=0)


def test_build(monkeypatch):
    with pytest.raises(adc.ADCError):
        adc.ADC.build(dev='BADDEVICE')

    monkeypatch.setattr(serial, "Serial", lambda *x, **y: mocks.SerialMock())
    adc.ADC.build(wait=0)


def test_acquire():

    ad = adc.ADC(mocks.SerialMock(), wait=0)

    ad.flush_input()

    with pytest.raises(adc.ADCError):
        ad.acquire(1, ch0=False, ch1=False)

    with pytest.raises(adc.ADCError):
        ad.acquire(0)

    SAMPLES = 50
    res = ad.acquire(SAMPLES)

    for ch in ad.AVAILABLE_CHANNELS:
        assert ch in res
        assert len(res[ch]) == SAMPLES

        for v in res[ch]:
            assert isinstance(v, float)
            assert 0 <= v <= ad.max_V

    res = ad.acquire(1, in_bytes=False)

    ad.close()

    print(res['CH0'][0])
