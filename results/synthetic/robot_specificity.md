## Robot specificity (synthetic arms)

Held-out tracking error (RMS over the five body joints). Rows are the arms' datasets, columns are the locks.

| dataset | svla_so101_pickplace-fit | svla_so100_pickplace-fit | default |
|---|---|---|---|
| synthetic/svla_so101_pickplace | 1.07° | 2.38° | 11.67° |
| synthetic/svla_so100_pickplace | 2.16° | 0.40° | 9.30° |

| dataset | own lock | own | best other lock | other | margin | own wins |
|---|---|---|---|---|---|---|
| synthetic/svla_so101_pickplace | svla_so101_pickplace-fit | 1.07° | svla_so100_pickplace-fit | 2.38° | 2.23× | yes |
| synthetic/svla_so100_pickplace | svla_so100_pickplace-fit | 0.40° | svla_so101_pickplace-fit | 2.16° | 5.38× | yes |

### synthetic/svla_so101_pickplace

| parameters | held-out error |
|---|---|
| own lock | 1.07° |
| same arm, refit on 20 other episodes | 1.77° |
| own fit with svla_so100_pickplace-fit's kp/kv pinned | 2.10° |
| svla_so100_pickplace-fit lock | 2.38° |
| true parameters (synthetic) | 0.34° |

Refits of the same data:

| fit | kp | kv | damping | armature | frictionloss | latency_frames | held-out |
|---|---|---|---|---|---|---|---|
| lock | 16.87 | 0.3418 | 0.07127 | 0.003634 | 0.03126 | 4 | 1.07° |
| seed 1 | 687.6 | 4.495 | 0.3259 | 0.004564 | 0.002389 | 4 | 1.91° |
| seed 2 | 10 | 0.1027 | 0.1069 | 0.002345 | 0.001732 | 4 | 0.55° |
| same-arm control | 60.37 | 0.1933 | 0.01012 | 0.003382 | 1.935 | 4 | 1.77° |
| truth | 10.54 | 0.1362 | 0.01226 | 0.002835 | 0.0943 | 4 | 0.34° |
| spread | ×68.7 | ×43.8 | ×4.57 | ×1.95 | ×18.1 | ±0 |  |

How far each parameter can move before the training error rises by 0.1° (linearized at the lock):

| param | lock | alone | others refit | trade-off |
|---|---|---|---|---|
| kp | 16.87 | ×1.22 | ×1.51 | 2.1 |
| kv | 0.3418 | ×1.32 | unbounded | 126 |
| damping | 0.07127 | ×3.8 | unbounded | 126 |
| armature | 0.003634 | ×52.8 | ×113 | 1.19 |
| frictionloss | 0.03126 | ×83.2 | ×1.27e+03 | 1.62 |

Least constrained direction (condition number 9e+05): +0.97 damping, -0.23 kv.

Latency: 0 frames 10.26°, 1 frames 7.66°, 2 frames 5.01°, 3 frames 2.38°, 4 frames 1.05°, 5 frames 3.41°.

Path from own lock to svla_so100_pickplace-fit (held-out): 1.07° → 1.26° → 1.43° → 1.58° → 1.71° → 1.83° → 2.63° → 2.59° → 2.53° → 2.46° → 2.38°.

### synthetic/svla_so100_pickplace

| parameters | held-out error |
|---|---|
| own lock | 0.40° |
| same arm, refit on 20 other episodes | 0.40° |
| own fit with svla_so101_pickplace-fit's kp/kv pinned | 1.17° |
| svla_so101_pickplace-fit lock | 2.16° |
| true parameters (synthetic) | 0.35° |

Refits of the same data:

| fit | kp | kv | damping | armature | frictionloss | latency_frames | held-out |
|---|---|---|---|---|---|---|---|
| lock | 182.1 | 5.785 | 0.2202 | 0.01013 | 0.001743 | 3 | 0.40° |
| seed 1 | 1215 | 49.62 | 0.03969 | 0.01646 | 0.01931 | 3 | 0.80° |
| seed 2 | 241.7 | 0.5577 | 0.01503 | 0.02084 | 0.03066 | 4 | 0.55° |
| same-arm control | 121.4 | 3.896 | 0.2629 | 0.01043 | 0.001157 | 3 | 0.40° |
| truth | 1427 | 41.59 | 0.1279 | 0.013 | 0.3899 | 3 | 0.35° |
| spread | ×6.67 | ×89 | ×14.6 | ×2.06 | ×17.6 | ±1 |  |

How far each parameter can move before the training error rises by 0.1° (linearized at the lock):

| param | lock | alone | others refit | trade-off |
|---|---|---|---|---|
| kp | 182.1 | ×1.14 | ×1.58 | 3.61 |
| kv | 5.785 | ×1.12 | ×1.53 | 3.7 |
| damping | 0.2202 | ×1.75 | ×1.82 | 1.07 |
| armature | 0.01013 | ×1.81 | ×1.93 | 1.1 |
| frictionloss | 0.001743 | unbounded | unbounded | 1 |

Least constrained direction (condition number 2.25e+04): +1.00 frictionloss.

Latency: 0 frames 8.44°, 1 frames 5.67°, 2 frames 2.86°, 3 frames 0.41°, 4 frames 2.89°, 5 frames 5.68°.

Path from own lock to svla_so101_pickplace-fit (held-out): 0.40° → 0.43° → 0.51° → 0.60° → 0.68° → 2.01° → 1.97° → 1.95° → 1.97° → 2.03° → 2.16°.
