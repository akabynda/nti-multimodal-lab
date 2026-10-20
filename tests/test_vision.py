import numpy as np
import pytest
from nti_lab.vision import pos_rppg, dominant_rate, finite_json

def test_pos_recovers_known_pulse_with_common_illumination():
    fs=25
    t=np.arange(0,45,1/fs)
    pulse=np.sin(2*np.pi*1.2*t)
    light=1+0.08*np.sin(2*np.pi*.15*t)
    rgb=np.column_stack([(130+0.5*pulse)*light,(110+2*pulse)*light,(90+.2*pulse)*light])
    rate,q=dominant_rate(pos_rppg(rgb,fs),fs)
    assert abs(rate-72)<2
    assert q>.7

def test_missing_rgb_not_fabricated():
    with pytest.raises(ValueError):
        pos_rppg(np.full((100,3),np.nan),25)
    assert dominant_rate(np.zeros(1000),25)==(None,None)
    assert finite_json({"x":float("nan")})=={"x":None}
