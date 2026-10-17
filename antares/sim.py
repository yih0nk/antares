"""MuJoCo side: load the arm model, map dataset units to joint space, replay.

Planned (Saturday morning):
- load the SO-101 MJCF (MuJoCo Menagerie, or TheRobotStudio's export)
- per-joint affine map from normalized dataset units to radians, seeded
  from the MJCF joint ranges
- ``replay(episode, params)``: set the sim to the real initial state, feed
  the commanded actions through the position actuators at the dataset
  rate, and return simulated joint trajectories plus per-frame body poses
  for the web viewer
"""
