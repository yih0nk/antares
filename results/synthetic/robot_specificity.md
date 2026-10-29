## Robot specificity (synthetic arms)

Held-out tracking error (RMS over the five body joints). Rows are the arms' datasets, columns are the locks.
Each dataset replays on its own model: synthetic/svla_so101_pickplace on `so101`, synthetic/svla_so100_pickplace on `so101`.

| dataset | svla_so101_pickplace-fit | svla_so100_pickplace-fit | default |
|---|---|---|---|
| synthetic/svla_so101_pickplace | 1.62° | 3.55° | 11.67° |
| synthetic/svla_so100_pickplace | 2.64° | 0.87° | 7.97° |

| dataset | own lock | own | best other lock | other | margin | own wins |
|---|---|---|---|---|---|---|
| synthetic/svla_so101_pickplace | svla_so101_pickplace-fit | 1.62° | svla_so100_pickplace-fit | 3.55° | 2.19× | yes |
| synthetic/svla_so100_pickplace | svla_so100_pickplace-fit | 0.87° | svla_so101_pickplace-fit | 2.64° | 3.06× | yes |

### synthetic/svla_so101_pickplace

| parameters | held-out error |
|---|---|
| own lock | 1.62° |
| same arm, refit on 20 other episodes | 1.26° |
| own fit with svla_so100_pickplace-fit's kp/kv pinned | 1.40° |
| svla_so100_pickplace-fit lock | 3.55° |
| true parameters (synthetic) | 0.34° |

Refits of the same data:

| fit | kp | kv | damping | armature | frictionloss | latency_frames | held-out |
|---|---|---|---|---|---|---|---|
| lock | 48.68 | 0.1157 | 0.3329 | 0.0005465 | 0.7259 | 4 | 1.62° |
| seed 1 | 11.64 | 0.1193 | 0.08527 | 0.004244 | 0.007518 | 4 | 0.52° |
| seed 2 | 45.2 | 0.1095 | 0.1871 | 0.0001097 | 0.0154 | 4 | 1.71° |
| same-arm control | 16.94 | 0.1109 | 0.03316 | 0.00229 | 0.004063 | 4 | 1.26° |
| truth | 10.54 | 0.1362 | 0.01226 | 0.002835 | 0.0943 | 4 | 0.34° |
| spread | ×4.18 | ×1.09 | ×3.9 | ×38.7 | ×96.6 | ±0 |  |

How far each parameter can move before the training error rises by 0.1° (linearized at the lock):

| param | lock | alone | others refit | trade-off |
|---|---|---|---|---|
| kp | 48.68 | ×1.55 | ×3.33 | 2.77 |
| kv | 0.1157 | ×39.5 | ×2.27e+03 | 2.1 |
| damping | 0.3329 | ×1.96 | ×3.2 | 1.73 |
| armature | 0.0005465 | ×24.4 | ×50.4 | 1.23 |
| frictionloss | 0.7259 | ×2.04 | ×4.88 | 2.22 |

Least constrained direction (condition number 548): +0.98 kv, +0.11 kp.

Latency: 0 frames 11.03°, 1 frames 8.50°, 2 frames 5.92°, 3 frames 3.40°, 4 frames 1.62°, 5 frames 2.93°.

Path from own lock to svla_so100_pickplace-fit (held-out): 1.62° → 1.63° → 1.66° → 1.67° → 1.67° → 1.66° → 3.67° → 3.66° → 3.64° → 3.60° → 3.55°.

### synthetic/svla_so100_pickplace

| parameters | held-out error |
|---|---|
| own lock | 0.87° |
| same arm, refit on 20 other episodes | 0.76° |
| own fit with svla_so101_pickplace-fit's kp/kv pinned | 0.79° |
| svla_so101_pickplace-fit lock | 2.64° |
| true parameters (synthetic) | 0.35° |

Refits of the same data:

| fit | kp | kv | damping | armature | frictionloss | latency_frames | held-out |
|---|---|---|---|---|---|---|---|
| lock | 36.45 | 0.4119 | 0.03269 | 0.000457 | 0.001495 | 3 | 0.87° |
| seed 1 | 4233 | 37.73 | 0.2769 | 0.0208 | 0.0117 | 3 | 0.77° |
| seed 2 | 64.98 | 0.3417 | 0.01061 | 0.007996 | 1.533 | 3 | 0.59° |
| same-arm control | 263 | 1.192 | 0.03206 | 0.005817 | 1.726 | 3 | 0.76° |
| truth | 70.9 | 0.1896 | 0.14 | 0.02116 | 1.219 | 3 | 0.35° |
| spread | ×116 | ×110 | ×26.1 | ×45.5 | ×1.03e+03 | ±0 |  |

How far each parameter can move before the training error rises by 0.1° (linearized at the lock):

| param | lock | alone | others refit | trade-off |
|---|---|---|---|---|
| kp | 36.45 | ×1.17 | ×1.18 | 1.04 |
| kv | 0.4119 | ×1.12 | ×1.24 | 1.97 |
| damping | 0.03269 | ×1.13 | ×1.15 | 1.13 |
| armature | 0.000457 | ×1.16 | ×1.34 | 2.02 |
| frictionloss | 0.001495 | unbounded | unbounded | 1.01 |

Least constrained direction (condition number 5.76e+06): +1.00 frictionloss.

Latency: 0 frames 8.85°, 1 frames 6.04°, 2 frames 3.23°, 3 frames 1.04°, 4 frames 2.96°, 5 frames 5.76°.

Path from own lock to svla_so101_pickplace-fit (held-out): 0.87° → 0.85° → 0.86° → 0.88° → 0.89° → 2.53° → 2.52° → 2.50° → 2.48° → 2.50° → 2.64°.
