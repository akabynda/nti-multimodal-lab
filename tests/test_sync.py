import pytest
from nti_lab.sync import clock_from_anchors,to_session

def test_clock_offset_and_drift():
    c=clock_from_anchors([0,100,200],[3,103.01,203.02])
    assert c["offset_s"]==pytest.approx(3)
    assert c["scale"]==pytest.approx(1.0001)
    assert to_session(100,c["offset_s"],c["scale"])==pytest.approx(103.01)

def test_invalid_clock():
    with pytest.raises(ValueError):
        clock_from_anchors([1,1],[2,3])
    with pytest.raises(ValueError):
        to_session(5,scale=-1)
