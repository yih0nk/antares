## Calibration dance (sim-only)

Synthetic SO-101 arms with known parameters and 0.3 units of sensor noise, recorded twice: on the 50 real teleoperated pick-and-place command sequences of `lerobot/svla_so101_pickplace` (20 train, 10 held out) and on one 61.8 s calibration dance. Each training set is fitted with 4 CMA-ES seeds at 150 and 600 evaluations; every fit is scored on the same 10 held-out pick-and-place episodes. The dance's commanded fingertip stays 17 cm or more above the table, body joints within ±50 units, peak speed outside steps 2.3 rad/s. Nothing here was run on an arm.

### so101 lock

True parameters: kp 10.54, kv 0.1362, damping 0.01226, armature 0.002835, frictionloss 0.0943, latency_frames 4. Held-out error at the truth: 0.340°.

Seed spread (max/min over the seeds; latency: max-min frames), per training set and budget:

| param | pick-and-place, 150 | dance, 150 | dance + pick-and-place, 150 | pick-and-place, 600 | dance, 600 | dance + pick-and-place, 600 |
|---|---|---|---|---|---|---|
| kp | ×3.09 | ×86.1 | ×131 | ×1.06 | ×1.05 | ×1.05 |
| kv | ×4.97 | ×32.4 | ×10.6 | ×1.3 | ×1.29 | ×1.27 |
| damping | ×29.9 | ×29.2 | ×32 | ×18.9 | ×2.76 | ×1.65 |
| armature | ×217 | ×4.55 | ×38.5 | ×4.3 | ×1.2 | ×13.7 |
| frictionloss | ×20.5 | ×20.5 | ×177 | ×62.1 | ×1.6 | ×2.89 |
| latency_frames | ±1 | ±0 | ±0 | ±1 | ±0 | ±0 |

Worst error against the truth over the seeds (max of fit/true, true/fit):

| param | pick-and-place, 150 | dance, 150 | dance + pick-and-place, 150 | pick-and-place, 600 | dance, 600 | dance + pick-and-place, 600 |
|---|---|---|---|---|---|---|
| kp | ×2.97 | ×82.4 | ×126 | ×1.03 | ×1.04 | ×1.05 |
| kv | ×3.68 | ×23.8 | ×11.5 | ×1.32 | ×1.2 | ×1.35 |
| damping | ×28 | ×24 | ×40.1 | ×32.7 | ×2.28 | ×4.89 |
| armature | ×26.1 | ×3.76 | ×15.3 | ×4.33 | ×1.12 | ×14.2 |
| frictionloss | ×80.5 | ×6.85 | ×25.4 | ×62.6 | ×1.57 | ×2.78 |
| latency_frames | 1 fr | 0 fr | 0 fr | 1 fr | 0 fr | 0 fr |

Linearized at the truth: how far each parameter can move, the others refit, before the training error rises 0.1°:

| param | pick-and-place | dance | dance + pick-and-place |
|---|---|---|---|
| kp | ×1.37 | ×1.18 | ×1.26 |
| kv | ×575 | ×5.67 | ×13.3 |
| damping | unbounded | unbounded | unbounded |
| armature | ×7.02 | ×1.95 | ×2.88 |
| frictionloss | ×2.82 | ×2.8 | ×2.77 |
| condition number | 1.1e+05 | 1.23e+04 | 1.99e+04 |

Training error of the fits against the truth's own, and held-out pick-and-place error:

| trained on | evals | frames | train, truth | train, best | train, worst | held-out best | median | worst |
|---|---|---|---|---|---|---|---|---|
| pick-and-place | 150 | 4547 | 0.342° | 0.461° | 0.748° | 0.455° | 0.562° | 0.722° |
| dance | 150 | 1854 | 0.348° | 0.472° | 1.666° | 0.391° | 0.700° | 1.245° |
| dance + pick-and-place | 150 | 6401 | 0.344° | 0.489° | 1.340° | 0.450° | 0.963° | 1.059° |
| pick-and-place | 600 | 4547 | 0.342° | 0.342° | 0.472° | 0.340° | 0.342° | 0.467° |
| dance | 600 | 1854 | 0.348° | 0.349° | 0.384° | 0.340° | 0.345° | 0.372° |
| dance + pick-and-place | 600 | 6401 | 0.344° | 0.345° | 0.447° | 0.341° | 0.343° | 0.417° |

### so100 lock

True parameters: kp 70.9, kv 0.1896, damping 0.14, armature 0.02116, frictionloss 1.219, latency_frames 3. Held-out error at the truth: 0.343°.

Seed spread (max/min over the seeds; latency: max-min frames), per training set and budget:

| param | pick-and-place, 150 | dance, 150 | dance + pick-and-place, 150 | pick-and-place, 600 | dance, 600 | dance + pick-and-place, 600 |
|---|---|---|---|---|---|---|
| kp | ×17.2 | ×23.4 | ×30.5 | ×47.3 | ×15 | ×2.92 |
| kv | ×11 | ×19.4 | ×46.5 | ×45.7 | ×39.4 | ×3.39 |
| damping | ×34.8 | ×4.84 | ×1.4 | ×14.3 | ×1.74 | ×1.74 |
| armature | ×17 | ×2.72 | ×12.8 | ×6.04 | ×1.72 | ×1.5 |
| frictionloss | ×45.8 | ×1.24e+03 | ×66.2 | ×35.9 | ×175 | ×16.3 |
| latency_frames | ±0 | ±0 | ±0 | ±0 | ±0 | ±0 |

Worst error against the truth over the seeds (max of fit/true, true/fit):

| param | pick-and-place, 150 | dance, 150 | dance + pick-and-place, 150 | pick-and-place, 600 | dance, 600 | dance + pick-and-place, 600 |
|---|---|---|---|---|---|---|
| kp | ×38 | ×34.6 | ×38.4 | ×26.4 | ×15 | ×34 |
| kv | ×70.4 | ×49 | ×95.8 | ×125 | ×50.7 | ×127 |
| damping | ×6.45 | ×4.26 | ×5.82 | ×13.8 | ×1.94 | ×3.78 |
| armature | ×9.98 | ×2.49 | ×11.2 | ×2.95 | ×1.85 | ×1.61 |
| frictionloss | ×956 | ×954 | ×797 | ×318 | ×199 | ×250 |
| latency_frames | 0 fr | 0 fr | 0 fr | 0 fr | 0 fr | 0 fr |

Linearized at the truth: how far each parameter can move, the others refit, before the training error rises 0.1°:

| param | pick-and-place | dance | dance + pick-and-place |
|---|---|---|---|
| kp | ×2.88 | ×2.02 | ×1.99 |
| kv | ×1.36e+03 | ×451 | ×81.8 |
| damping | ×568 | ×7.17 | ×14.8 |
| armature | ×2.86 | ×2.02 | ×1.82 |
| frictionloss | ×3.03 | ×2.42 | ×2.22 |
| condition number | 2.08e+03 | 2e+03 | 666 |

Training error of the fits against the truth's own, and held-out pick-and-place error:

| trained on | evals | frames | train, truth | train, best | train, worst | held-out best | median | worst |
|---|---|---|---|---|---|---|---|---|
| pick-and-place | 150 | 4547 | 0.342° | 0.561° | 0.614° | 0.557° | 0.567° | 0.617° |
| dance | 150 | 1854 | 0.343° | 0.532° | 0.610° | 0.532° | 0.570° | 0.726° |
| dance + pick-and-place | 150 | 6401 | 0.342° | 0.596° | 0.661° | 0.587° | 0.611° | 0.636° |
| pick-and-place | 600 | 4547 | 0.342° | 0.523° | 0.538° | 0.516° | 0.535° | 0.537° |
| dance | 600 | 1854 | 0.343° | 0.346° | 0.495° | 0.349° | 0.578° | 0.777° |
| dance + pick-and-place | 600 | 6401 | 0.342° | 0.540° | 0.568° | 0.553° | 0.556° | 0.564° |

### stock MJCF

True parameters: kp 998.2, kv 2.731, damping 0.6, armature 0.028, frictionloss 0.052, latency_frames 2. Held-out error at the truth: 0.340°.

Seed spread (max/min over the seeds; latency: max-min frames), per training set and budget:

| param | pick-and-place, 150 | dance, 150 | dance + pick-and-place, 150 | pick-and-place, 600 | dance, 600 | dance + pick-and-place, 600 |
|---|---|---|---|---|---|---|
| kp | ×32.5 | ×40.3 | ×18.1 | ×4.72 | ×4.64 | ×1.17 |
| kv | ×75.3 | ×60.9 | ×3.78 | ×2.15 | ×5.28 | ×1.22 |
| damping | ×24.9 | ×2.27 | ×1.98 | ×1.83 | ×1.08 | ×1.33 |
| armature | ×2.24 | ×3.93 | ×25.6 | ×1.13 | ×1.09 | ×1.1 |
| frictionloss | ×14.3 | ×4.36 | ×311 | ×8.78 | ×236 | ×175 |
| latency_frames | ±0 | ±1 | ±0 | ±0 | ±0 | ±0 |

Worst error against the truth over the seeds (max of fit/true, true/fit):

| param | pick-and-place, 150 | dance, 150 | dance + pick-and-place, 150 | pick-and-place, 600 | dance, 600 | dance + pick-and-place, 600 |
|---|---|---|---|---|---|---|
| kp | ×6.49 | ×8.49 | ×4.55 | ×5.53 | ×4.12 | ×1.19 |
| kv | ×16.4 | ×14.3 | ×3.7 | ×2.34 | ×5.12 | ×1.25 |
| damping | ×56.6 | ×1.65 | ×1.59 | ×1.82 | ×1.06 | ×1.3 |
| armature | ×1.65 | ×2.31 | ×47 | ×1.1 | ×1.05 | ×1.08 |
| frictionloss | ×7.99 | ×12.8 | ×19.3 | ×10.3 | ×48.4 | ×18.9 |
| latency_frames | 0 fr | 1 fr | 0 fr | 0 fr | 0 fr | 0 fr |

Linearized at the truth: how far each parameter can move, the others refit, before the training error rises 0.1°:

| param | pick-and-place | dance | dance + pick-and-place |
|---|---|---|---|
| kp | ×1.65 | ×3.07 | ×1.76 |
| kv | ×1.68 | ×3.18 | ×1.8 |
| damping | ×1.46 | ×1.42 | ×1.43 |
| armature | ×1.6 | ×1.46 | ×1.51 |
| frictionloss | ×2.18 | ×14.2 | ×2.48 |
| condition number | 3.89 | 127 | 7.94 |

Training error of the fits against the truth's own, and held-out pick-and-place error:

| trained on | evals | frames | train, truth | train, best | train, worst | held-out best | median | worst |
|---|---|---|---|---|---|---|---|---|
| pick-and-place | 150 | 4547 | 0.341° | 0.388° | 0.490° | 0.391° | 0.406° | 0.485° |
| dance | 150 | 1854 | 0.341° | 0.397° | 0.468° | 0.377° | 0.501° | 0.563° |
| dance + pick-and-place | 150 | 6401 | 0.341° | 0.404° | 0.519° | 0.391° | 0.436° | 0.480° |
| pick-and-place | 600 | 4547 | 0.341° | 0.346° | 0.367° | 0.345° | 0.349° | 0.366° |
| dance | 600 | 1854 | 0.341° | 0.341° | 0.347° | 0.345° | 0.357° | 0.371° |
| dance + pick-and-place | 600 | 6401 | 0.341° | 0.345° | 0.350° | 0.345° | 0.346° | 0.350° |

