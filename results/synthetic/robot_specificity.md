## Robot specificity (synthetic arms)

Held-out tracking error (RMS over the five body joints). Rows are the arms' datasets, columns are the locks.

| dataset | svla_so101_pickplace-fit | svla_so100_pickplace-fit | default |
|---|---|---|---|
| synthetic/svla_so101_pickplace | 0.54° | 0.79° | 11.33° |
| synthetic/svla_so100_pickplace | 0.79° | 0.61° | 9.70° |

| dataset | own lock | own | best other lock | other | margin | own wins |
|---|---|---|---|---|---|---|
| synthetic/svla_so101_pickplace | svla_so101_pickplace-fit | 0.54° | svla_so100_pickplace-fit | 0.79° | 1.46× | yes |
| synthetic/svla_so100_pickplace | svla_so100_pickplace-fit | 0.61° | svla_so101_pickplace-fit | 0.79° | 1.29× | yes |

### synthetic/svla_so101_pickplace

| parameters | held-out error |
|---|---|
| own lock | 0.54° |
| same arm, refit on 20 other episodes | 0.48° |
| own fit with svla_so100_pickplace-fit's kp/kv pinned | 0.68° |
| svla_so100_pickplace-fit lock | 0.79° |
| true parameters (synthetic) | 0.34° |

Refits of the same data:

| fit | kp | kv | damping | armature | frictionloss | latency_frames | held-out |
|---|---|---|---|---|---|---|---|
| lock | 431.6 | 1.963 | 0.2977 | 0.003261 | 0.001001 | 4 | 0.54° |
| seed 1 | 3658 | 26.78 | 0.07668 | 0.01673 | 0.8805 | 4 | 0.52° |
| seed 2 | 3176 | 31.05 | 0.3495 | 0.02199 | 0.0277 | 4 | 0.43° |
| same-arm control | 3315 | 49.87 | 0.01419 | 0.01922 | 0.009789 | 4 | 0.48° |
| truth | 93.7 | 1.572 | 0.05272 | 0.0197 | 0.02518 | 4 | 0.34° |
| spread | ×8.47 | ×15.8 | ×4.56 | ×6.74 | ×880 | ±0 |  |

How far each parameter can move before the training error rises by 0.1° (linearized at the lock):

| param | lock | alone | others refit | trade-off |
|---|---|---|---|---|
| kp | 431.6 | ×1.27 | ×1.29 | 1.09 |
| kv | 1.963 | ×1.35 | ×1.4 | 1.13 |
| damping | 0.2977 | ×1.39 | ×1.39 | 1 |
| armature | 0.003261 | ×1.49 | ×1.53 | 1.05 |
| frictionloss | 0.001001 | ×674 | ×675 | 1 |

Least constrained direction (condition number 491): +1.00 frictionloss.

Latency: 0 frames 10.71°, 1 frames 8.18°, 2 frames 5.57°, 3 frames 2.93°, 4 frames 0.51°, 5 frames 2.53°.

Path from own lock to svla_so100_pickplace-fit (held-out): 0.54° → 0.55° → 0.57° → 0.59° → 0.63° → 0.66° → 0.70° → 0.74° → 0.76° → 0.78° → 0.79°.

### synthetic/svla_so100_pickplace

| parameters | held-out error |
|---|---|
| own lock | 0.61° |
| same arm, refit on 20 other episodes | 0.60° |
| own fit with svla_so101_pickplace-fit's kp/kv pinned | 0.58° |
| svla_so101_pickplace-fit lock | 0.79° |
| true parameters (synthetic) | 0.35° |

Refits of the same data:

| fit | kp | kv | damping | armature | frictionloss | latency_frames | held-out |
|---|---|---|---|---|---|---|---|
| lock | 1865 | 6.623 | 0.3853 | 0.03026 | 0.1447 | 4 | 0.61° |
| seed 1 | 764.3 | 28.32 | 0.01495 | 0.02196 | 0.006108 | 3 | 0.41° |
| seed 2 | 182.8 | 0.6691 | 0.1752 | 0.01579 | 0.6912 | 4 | 0.56° |
| same-arm control | 537.6 | 2.471 | 0.2162 | 0.01398 | 0.007179 | 4 | 0.60° |
| truth | 1246 | 42.42 | 0.1854 | 0.01376 | 0.622 | 3 | 0.35° |
| spread | ×10.2 | ×42.3 | ×25.8 | ×1.92 | ×113 | ±1 |  |

How far each parameter can move before the training error rises by 0.1° (linearized at the lock):

| param | lock | alone | others refit | trade-off |
|---|---|---|---|---|
| kp | 1865 | ×1.15 | ×1.15 | 1.01 |
| kv | 6.623 | ×1.22 | ×1.22 | 1.01 |
| damping | 0.3853 | ×1.17 | ×1.18 | 1.06 |
| armature | 0.03026 | ×1.19 | ×1.21 | 1.07 |
| frictionloss | 0.1447 | ×1.24 | ×1.24 | 1.03 |

Least constrained direction (condition number 3.21): +0.90 kv, +0.38 kp, +0.15 armature, -0.15 damping.

Latency: 0 frames 11.05°, 1 frames 8.33°, 2 frames 5.54°, 3 frames 2.73°, 4 frames 0.73°, 5 frames 3.17°.

Path from own lock to svla_so101_pickplace-fit (held-out): 0.61° → 0.60° → 0.61° → 0.62° → 0.64° → 0.68° → 0.71° → 0.74° → 0.76° → 0.78° → 0.79°.
