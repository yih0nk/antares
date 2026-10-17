"""Fit simulator parameters to one robot's logs.

Planned (Saturday afternoon): CMA-ES over a small vector (actuator gain,
damping, armature, friction loss, action latency), objective = mean
tracking error on training episodes, scored on held-out episodes. Emits
progress events so the web UI can stream the fit live.
"""
