import pytest

import pyvisa

from rapp.pm100 import PM100, PM100Error


def test_PM100(monkeypatch):
    pm100 = PM100.build("mock_resource", duration=10, mock=True)
    assert isinstance(pm100.get_power(), float)

    # Raises error if getconfigure is not POW and ask power.
    pm100._detector.getconfigure = "NOTPOW"
    with pytest.raises(PM100Error):
        pm100.get_power()

    pm100.start_measurement()
    assert isinstance(pm100.fetch_measurement(), float)

    pm100.close()

    # Cool library to mock VISA resources.
    # https://pyvisa.readthedocs.io/projects/pyvisa-sim/en/latest/
    sim_rm = pyvisa.ResourceManager('@sim')
    pm100 = PM100.build("GPIB0::8::INSTR", rm=sim_rm, mock=False)

    with pytest.raises(PM100Error):
        pm100 = PM100.build("notfound", rm=sim_rm, mock=False)

    pm100.start_measurement(duration=3)
    pm100.fetch_measurement(tolerable_fetch_time=0)

    monkeypatch.setattr(pyvisa, "ResourceManager", lambda *x, **y: sim_rm)
    pm100 = PM100.build("GPIB0::8::INSTR", mock=False)
