## Robot specificity

Held-out tracking error (RMS over the five body joints). Rows are the arms' datasets, columns are the locks.

| dataset | svla_so101_pickplace | svla_so100_pickplace | default |
|---|---|---|---|
| lerobot/svla_so101_pickplace | 1.05° | 1.14° | 6.01° |
| lerobot/svla_so100_pickplace | 0.85° | 0.79° | 2.74° |

| dataset | own lock | own | best other lock | other | margin | own wins |
|---|---|---|---|---|---|---|
| lerobot/svla_so101_pickplace | svla_so101_pickplace | 1.05° | svla_so100_pickplace | 1.14° | 1.08× | yes |
| lerobot/svla_so100_pickplace | svla_so100_pickplace | 0.79° | svla_so101_pickplace | 0.85° | 1.07× | yes |

### lerobot/svla_so101_pickplace

| parameters | held-out error |
|---|---|
| own lock | 1.05° |
| same arm, refit on 20 other episodes | 1.05° |
| own fit with svla_so100_pickplace's kp/kv pinned | 1.05° |
| svla_so100_pickplace lock | 1.14° |

Refits of the same data:

| fit | kp | kv | damping | armature | frictionloss | latency_frames | held-out |
|---|---|---|---|---|---|---|---|
| lock | 93.7 | 1.572 | 0.05272 | 0.0197 | 0.02518 | 4 | 1.05° |
| seed 1 | 50.02 | 0.6134 | 0.04696 | 0.00131 | 0.07742 | 4 | 1.00° |
| seed 2 | 103.5 | 4.578 | 0.712 | 0.02056 | 0.005809 | 3 | 1.01° |
| same-arm control | 54.09 | 2.072 | 0.3138 | 0.001614 | 0.02014 | 3 | 1.05° |
| spread | ×2.07 | ×7.46 | ×15.2 | ×15.7 | ×13.3 | ±1 |  |

How far each parameter can move before the training error rises by 0.1° (linearized at the lock):

| param | lock | alone | others refit | trade-off |
|---|---|---|---|---|
| kp | 93.7 | ×1.92 | ×8.85 | 3.34 |
| kv | 1.572 | ×1.85 | ×9.18 | 3.61 |
| damping | 0.05272 | ×2.31e+04 | ×6.66e+05 | 1.33 |
| armature | 0.0197 | ×73.6 | ×161 | 1.18 |
| frictionloss | 0.02518 | unbounded | unbounded | 1.14 |

Least constrained direction (condition number 1.62e+03): +0.96 frictionloss, -0.28 damping.

Latency: 0 frames 5.90°, 1 frames 4.52°, 2 frames 3.14°, 3 frames 1.82°, 4 frames 1.06°, 5 frames 1.81°.

Path from own lock to svla_so100_pickplace (held-out): 1.05° → 1.07° → 1.08° → 1.11° → 1.15° → 1.19° → 1.37° → 1.32° → 1.26° → 1.21° → 1.14°.

### lerobot/svla_so100_pickplace

| parameters | held-out error |
|---|---|
| own lock | 0.79° |
| same arm, refit on 20 other episodes | 0.79° |
| own fit with svla_so101_pickplace's kp/kv pinned | 0.73° |
| svla_so101_pickplace lock | 0.85° |

Refits of the same data:

| fit | kp | kv | damping | armature | frictionloss | latency_frames | held-out |
|---|---|---|---|---|---|---|---|
| lock | 1246 | 42.42 | 0.1854 | 0.01376 | 0.622 | 3 | 0.79° |
| seed 1 | 55.98 | 1.086 | 0.08484 | 0.00391 | 0.01685 | 3 | 0.74° |
| seed 2 | 661.4 | 39.63 | 0.4923 | 0.03245 | 0.006188 | 2 | 0.78° |
| same-arm control | 1603 | 48.83 | 0.1357 | 0.01663 | 0.09388 | 3 | 0.79° |
| spread | ×22.3 | ×39 | ×5.8 | ×8.3 | ×101 | ±1 |  |

How far each parameter can move before the training error rises by 0.1° (linearized at the lock):

| param | lock | alone | others refit | trade-off |
|---|---|---|---|---|
| kp | 1246 | ×1.26 | ×1.3 | 1.12 |
| kv | 42.42 | ×1.34 | ×1.4 | 1.14 |
| damping | 0.1854 | ×1.41 | ×1.42 | 1.02 |
| armature | 0.01376 | ×1.55 | ×1.57 | 1.02 |
| frictionloss | 0.622 | ×1.52 | ×1.54 | 1.02 |

Least constrained direction (condition number 2.88): +0.70 kp, +0.67 kv, +0.23 armature.

Latency: 0 frames 2.14°, 1 frames 1.51°, 2 frames 0.96°, 3 frames 0.77°, 4 frames 1.12°, 5 frames 1.69°.

Path from own lock to svla_so101_pickplace (held-out): 0.79° → 0.79° → 0.80° → 0.81° → 0.81° → 0.98° → 0.95° → 0.93° → 0.90° → 0.87° → 0.85°.
