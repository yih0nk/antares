## Robot specificity

Held-out tracking error (RMS over the five body joints). Rows are the arms' datasets, columns are the locks.

| dataset | svla_so101_pickplace | svla_so100_pickplace | default |
|---|---|---|---|
| lerobot/svla_so101_pickplace | 0.74° | 1.27° | 6.01° |
| lerobot/svla_so100_pickplace | 0.96° | 0.79° | 2.74° |

| dataset | own lock | own | best other lock | other | margin | own wins |
|---|---|---|---|---|---|---|
| lerobot/svla_so101_pickplace | svla_so101_pickplace | 0.74° | svla_so100_pickplace | 1.27° | 1.73× | yes |
| lerobot/svla_so100_pickplace | svla_so100_pickplace | 0.79° | svla_so101_pickplace | 0.96° | 1.22× | yes |

### lerobot/svla_so101_pickplace

| parameters | held-out error |
|---|---|
| own lock | 0.74° |
| same arm, refit on 20 other episodes | 1.05° |
| own fit with svla_so100_pickplace's kp/kv pinned | 1.09° |
| svla_so100_pickplace lock | 1.27° |

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

Path from own lock to svla_so100_pickplace (held-out): 0.74° → 0.80° → 0.90° → 0.98° → 1.03° → 1.07° → 1.56° → 1.48° → 1.40° → 1.33° → 1.27°.

### lerobot/svla_so100_pickplace

| parameters | held-out error |
|---|---|
| own lock | 0.79° |
| same arm, refit on 20 other episodes | 0.80° |
| own fit with svla_so101_pickplace's kp/kv pinned | 0.65° |
| svla_so101_pickplace lock | 0.96° |

Refits of the same data:

| fit | kp | kv | damping | armature | frictionloss | latency_frames | held-out |
|---|---|---|---|---|---|---|---|
| lock | 1427 | 41.59 | 0.1279 | 0.013 | 0.3899 | 3 | 0.79° |
| seed 1 | 14.41 | 0.3328 | 0.8723 | 0.000103 | 0.1662 | 1 | 0.57° |
| seed 2 | 12.62 | 0.5152 | 0.2098 | 0.003005 | 0.2346 | 2 | 0.66° |
| same-arm control | 1701 | 48.3 | 0.01306 | 0.03245 | 0.03725 | 3 | 0.80° |
| spread | ×113 | ×125 | ×6.82 | ×126 | ×2.35 | ±2 |  |

How far each parameter can move before the training error rises by 0.1° (linearized at the lock):

| param | lock | alone | others refit | trade-off |
|---|---|---|---|---|
| kp | 1427 | ×1.36 | ×1.41 | 1.12 |
| kv | 41.59 | ×1.37 | ×1.42 | 1.12 |
| damping | 0.1279 | ×1.45 | ×1.46 | 1.01 |
| armature | 0.013 | ×1.56 | ×1.58 | 1.02 |
| frictionloss | 0.3899 | ×1.58 | ×1.6 | 1.02 |

Least constrained direction (condition number 2.73): +0.72 kv, +0.68 kp, +0.16 frictionloss.

Latency: 0 frames 2.25°, 1 frames 1.61°, 2 frames 1.04°, 3 frames 0.77°, 4 frames 1.05°, 5 frames 1.60°.

Path from own lock to svla_so101_pickplace (held-out): 0.79° → 0.80° → 0.81° → 0.82° → 0.82° → 0.90° → 0.86° → 0.81° → 0.76° → 0.76° → 0.96°.
