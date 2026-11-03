## Robot specificity

Held-out tracking error (RMS over the five body joints). Rows are the arms' datasets, columns are the locks.
Each dataset replays on its own model: lerobot/svla_so101_pickplace on `so101`, lerobot/svla_so100_pickplace on `so101_deg`.

| dataset | svla_so101_pickplace | svla_so100_pickplace | default |
|---|---|---|---|
| lerobot/svla_so101_pickplace | 0.74° | 1.83° | 6.01° |
| lerobot/svla_so100_pickplace | 0.92° | 0.77° | 2.61° |

| dataset | own lock | own | best other lock | other | margin | own wins |
|---|---|---|---|---|---|---|
| lerobot/svla_so101_pickplace | svla_so101_pickplace | 0.74° | svla_so100_pickplace | 1.83° | 2.48× | yes |
| lerobot/svla_so100_pickplace | svla_so100_pickplace | 0.77° | svla_so101_pickplace | 0.92° | 1.20× | yes |

### lerobot/svla_so101_pickplace

| parameters | held-out error |
|---|---|
| own lock | 0.74° |
| same arm, refit on 20 other episodes | 1.05° |
| own fit with svla_so100_pickplace's kp/kv pinned | 1.02° |
| svla_so100_pickplace lock | 1.83° |

Refits of the same data:

| fit | kp | kv | damping | armature | frictionloss | latency_frames | held-out |
|---|---|---|---|---|---|---|---|
| lock | 10.54 | 0.1362 | 0.01226 | 0.002835 | 0.0943 | 4 | 0.74° |
| seed 1 | 27.11 | 0.2116 | 0.05859 | 0.002782 | 0.3733 | 4 | 0.91° |
| seed 2 | 259.7 | 13.6 | 0.01928 | 0.005788 | 0.03272 | 3 | 1.06° |
| same-arm control | 410.7 | 22.07 | 0.01227 | 0.01133 | 0.003153 | 3 | 1.05° |
| spread | ×24.6 | ×99.9 | ×4.78 | ×2.08 | ×11.4 | ±1 |  |

How far each parameter can move before the training error rises by 0.1° (linearized at the lock):

| param | lock | alone | others refit | trade-off |
|---|---|---|---|---|
| kp | 10.54 | ×1.35 | ×1.57 | 1.51 |
| kv | 0.1362 | ×1.88 | ×8.8e+03 | 14.4 |
| damping | 0.01226 | ×1.11e+03 | unbounded | 14.4 |
| armature | 0.002835 | ×12.5 | ×15.3 | 1.08 |
| frictionloss | 0.0943 | ×2.72 | ×4.27 | 1.45 |

Least constrained direction (condition number 1.1e+05): +1.00 damping.

Latency: 0 frames 5.82°, 1 frames 4.42°, 2 frames 3.00°, 3 frames 1.61°, 4 frames 0.75°, 5 frames 1.71°.

Path from own lock to svla_so100_pickplace (held-out): 0.74° → 0.76° → 0.80° → 0.85° → 0.90° → 0.94° → 1.84° → 1.86° → 1.87° → 1.86° → 1.83°.

### lerobot/svla_so100_pickplace

| parameters | held-out error |
|---|---|
| own lock | 0.77° |
| same arm, refit on 20 other episodes | 0.78° |
| own fit with svla_so101_pickplace's kp/kv pinned | 0.65° |
| svla_so101_pickplace lock | 0.92° |

Refits of the same data:

| fit | kp | kv | damping | armature | frictionloss | latency_frames | held-out |
|---|---|---|---|---|---|---|---|
| lock | 70.9 | 0.1896 | 0.14 | 0.02116 | 1.219 | 3 | 0.77° |
| seed 1 | 17.04 | 0.6296 | 0.0525 | 0.0004562 | 0.1186 | 3 | 0.60° |
| seed 2 | 99.03 | 6.875 | 0.1053 | 0.02187 | 0.0196 | 2 | 0.71° |
| same-arm control | 1085 | 38.99 | 0.04845 | 0.01282 | 0.3943 | 3 | 0.78° |
| spread | ×5.81 | ×36.3 | ×2.67 | ×48 | ×62.2 | ±1 |  |

How far each parameter can move before the training error rises by 0.1° (linearized at the lock):

| param | lock | alone | others refit | trade-off |
|---|---|---|---|---|
| kp | 70.9 | ×1.58 | ×9.11 | 4.84 |
| kv | 0.1896 | ×220 | unbounded | 5.99 |
| damping | 0.14 | ×836 | unbounded | 5.47 |
| armature | 0.02116 | ×4.38 | ×15.8 | 1.87 |
| frictionloss | 1.219 | ×1.72 | ×9.89 | 4.23 |

Least constrained direction (condition number 2.03e+04): +0.72 damping, -0.70 kv.

Latency: 0 frames 2.21°, 1 frames 1.61°, 2 frames 1.06°, 3 frames 0.75°, 4 frames 0.94°, 5 frames 1.44°.

Path from own lock to svla_so101_pickplace (held-out): 0.77° → 0.74° → 0.72° → 0.69° → 0.66° → 0.77° → 0.75° → 0.74° → 0.76° → 0.81° → 0.92°.
