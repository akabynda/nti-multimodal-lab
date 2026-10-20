"""Explicit clocks: session_time = offset + scale * source_time."""
import numpy as np


def clock_from_anchors(source, session):
    x,y=np.asarray(source,float),np.asarray(session,float)
    if len(x)!=len(y) or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite paired synchronization anchors required")
    if len(x)==1:
        return {"offset_s":float(y[0]-x[0]),"scale":1.0,"residual_s":0.0}
    if np.ptp(x)<1e-9:
        raise ValueError("Anchors must span time")
    scale,offset=np.polyfit(x,y,1)
    if scale<=0:
        raise ValueError("Clock must be monotonic")
    return {"offset_s":float(offset),"scale":float(scale),
            "residual_s":float(np.sqrt(np.mean((y-(offset+scale*x))**2)))}


def to_session(t, offset_s=0., scale=1.):
    if not np.isfinite(scale) or scale<=0 or not np.isfinite(offset_s):
        raise ValueError("Invalid clock")
    return np.asarray(t)*scale+offset_s
