## Does calibration help a policy transfer? (sim-to-sim)

Joint-space reaching on a hidden MuJoCo arm. Success = all five body joints within 2 deg of the target for the last 5 of 60 frames at 30 fps; final error = mean absolute joint error over those frames. 3 training seed(s) per variant (0, 1, 2), 24 training tasks, 64 held-out tasks (seed 2024), 150 CMA-ES generations, hidden size 6 (111 policy parameters). Values are mean +- sd over seeds.

| variant | hidden arm | success in train sim | success on hidden arm | final error on hidden arm (deg) |
|---|---|---|---|---|
| default sim | so101 | 100% +- 0 | 20% +- 0 | 0.98 +- 0.00 |
| default sim | so100 | 100% +- 0 | 100% +- 0 | 0.01 +- 0.00 |
| fitted lock | so101 | 89% +- 3 | 86% +- 2 | 0.58 +- 0.05 |
| fitted lock | so100 | 89% +- 2 | 100% +- 0 | 0.32 +- 0.27 |
| fitted lock + data-derived DR | so101 | 86% +- 2 | 82% +- 2 | 0.61 +- 0.08 |
| fitted lock + data-derived DR | so100 | 34% +- 12 | 74% +- 25 | 0.77 +- 0.25 |
| oracle (true parameters) | so101 | 83% +- 7 | 83% +- 7 | 0.67 +- 0.18 |
| oracle (true parameters) | so100 | 100% +- 0 | 100% +- 0 | 0.01 +- 0.00 |
| default sim + hand-picked +-30% DR | so101 | 100% +- 0 | 29% +- 2 | 1.66 +- 0.65 |
| default sim + hand-picked +-30% DR | so100 | 100% +- 0 | 100% +- 0 | 0.14 +- 0.05 |

### Hidden arm `so101`

Lock fitted to 20 synthesized episodes (300 frames, noise 0.3 units), best of 3 CMA-ES seed(s) x 150 evaluations; held-out tracking RMSE: default 8.34 deg, fitted 0.52 deg, true parameters 0.34 deg. DR ranges from 8 refits x 12 episodes x 60 evaluations.

| | kp | kv | damping | armature | frictionloss | latency_frames |
|---|---|---|---|---|---|---|
| truth | 10.54 | 0.1362 | 0.01226 | 0.002835 | 0.0943 | 4 |
| fitted | 12.1 | 0.1846 | 0.03652 | 0.0008414 | 0.00819 | 4 |
| DR lo | 10.12 | 0.1005 | 0.01077 | 0.0003819 | 0.007657 | 4 |
| DR hi | 12.1 | 0.2057 | 0.1018 | 0.003541 | 0.05525 | 4 |

Untrained policy (`u = target`) on the hidden arm: success 20% [11, 31], final error 0.98 deg.

| variant | seed | success in train sim [95% CI] | success on hidden arm [95% CI] | final error on hidden arm (deg) [95% CI] | CMA-ES evaluations | train time (s) |
|---|---|---|---|---|---|---|
| default sim | 0 | 100% [100, 100] | 20% [11, 31] | 0.98 [0.90, 1.06] | 2701 | 284 |
| default sim | 1 | 100% [100, 100] | 20% [11, 31] | 0.98 [0.90, 1.06] | 2701 | 280 |
| default sim | 2 | 100% [100, 100] | 20% [11, 31] | 0.98 [0.90, 1.06] | 2701 | 280 |
| fitted lock | 0 | 86% [77, 94] | 89% [81, 95] | 0.55 [0.46, 0.66] | 2701 | 275 |
| fitted lock | 1 | 89% [81, 95] | 84% [75, 92] | 0.64 [0.56, 0.72] | 2701 | 272 |
| fitted lock | 2 | 92% [86, 98] | 86% [77, 94] | 0.55 [0.48, 0.62] | 2701 | 274 |
| fitted lock + data-derived DR | 0 | 86% [77, 94] | 83% [73, 92] | 0.56 [0.50, 0.64] | 2700 | 276 |
| fitted lock + data-derived DR | 1 | 88% [80, 95] | 80% [70, 89] | 0.70 [0.62, 0.79] | 2700 | 277 |
| fitted lock + data-derived DR | 2 | 84% [75, 92] | 84% [75, 92] | 0.56 [0.50, 0.63] | 2700 | 277 |
| oracle (true parameters) | 0 | 78% [67, 88] | 78% [67, 88] | 0.85 [0.75, 0.97] | 2701 | 269 |
| oracle (true parameters) | 1 | 81% [70, 91] | 81% [70, 91] | 0.68 [0.59, 0.77] | 2701 | 296 |
| oracle (true parameters) | 2 | 91% [83, 97] | 91% [83, 97] | 0.49 [0.43, 0.56] | 2701 | 293 |
| default sim + hand-picked +-30% DR | 0 | 100% [100, 100] | 27% [17, 38] | 2.28 [1.85, 2.70] | 2700 | 319 |
| default sim + hand-picked +-30% DR | 1 | 100% [100, 100] | 31% [20, 44] | 1.71 [1.33, 2.10] | 2700 | 322 |
| default sim + hand-picked +-30% DR | 2 | 100% [100, 100] | 28% [19, 39] | 0.98 [0.89, 1.06] | 2700 | 316 |

### Hidden arm `so100`

Lock fitted to 20 synthesized episodes (300 frames, noise 0.3 units), best of 3 CMA-ES seed(s) x 150 evaluations; held-out tracking RMSE: default 6.32 deg, fitted 0.43 deg, true parameters 0.39 deg. DR ranges from 8 refits x 12 episodes x 60 evaluations.

| | kp | kv | damping | armature | frictionloss | latency_frames |
|---|---|---|---|---|---|---|
| truth | 1427 | 41.59 | 0.1279 | 0.013 | 0.3899 | 3 |
| fitted | 119.7 | 0.232 | 0.06971 | 0.01593 | 0.001319 | 4 |
| DR lo | 119.7 | 0.1757 | 0.02661 | 0.01593 | 0.001081 | 4 |
| DR hi | 176 | 0.3118 | 0.1346 | 0.02174 | 0.002381 | 4 |

Untrained policy (`u = target`) on the hidden arm: success 100% [100, 100], final error 0.01 deg.

| variant | seed | success in train sim [95% CI] | success on hidden arm [95% CI] | final error on hidden arm (deg) [95% CI] | CMA-ES evaluations | train time (s) |
|---|---|---|---|---|---|---|
| default sim | 0 | 100% [100, 100] | 100% [100, 100] | 0.01 [0.01, 0.01] | 2701 | 311 |
| default sim | 1 | 100% [100, 100] | 100% [100, 100] | 0.01 [0.01, 0.01] | 2701 | 314 |
| default sim | 2 | 100% [100, 100] | 100% [100, 100] | 0.01 [0.01, 0.01] | 2701 | 312 |
| fitted lock | 0 | 88% [78, 95] | 100% [100, 100] | 0.01 [0.01, 0.01] | 2701 | 293 |
| fitted lock | 1 | 91% [83, 97] | 100% [100, 100] | 0.42 [0.39, 0.45] | 2701 | 295 |
| fitted lock | 2 | 88% [80, 94] | 100% [100, 100] | 0.52 [0.48, 0.57] | 2701 | 353 |
| fitted lock + data-derived DR | 0 | 20% [11, 31] | 45% [34, 58] | 1.06 [0.97, 1.13] | 2700 | 363 |
| fitted lock + data-derived DR | 1 | 44% [33, 56] | 89% [81, 95] | 0.66 [0.60, 0.73] | 2700 | 359 |
| fitted lock + data-derived DR | 2 | 39% [28, 52] | 89% [81, 95] | 0.58 [0.54, 0.63] | 2700 | 359 |
| oracle (true parameters) | 0 | 100% [100, 100] | 100% [100, 100] | 0.01 [0.01, 0.01] | 2701 | 360 |
| oracle (true parameters) | 1 | 100% [100, 100] | 100% [100, 100] | 0.01 [0.01, 0.01] | 2701 | 357 |
| oracle (true parameters) | 2 | 100% [100, 100] | 100% [100, 100] | 0.01 [0.01, 0.01] | 2701 | 356 |
| default sim + hand-picked +-30% DR | 0 | 100% [100, 100] | 100% [100, 100] | 0.19 [0.17, 0.21] | 2700 | 364 |
| default sim + hand-picked +-30% DR | 1 | 100% [100, 100] | 100% [100, 100] | 0.13 [0.11, 0.14] | 2700 | 364 |
| default sim + hand-picked +-30% DR | 2 | 100% [100, 100] | 100% [100, 100] | 0.09 [0.08, 0.10] | 2700 | 361 |

Run: antares 0.0.1, 2026-10-18T02:37:33Z, 19.4 min wall clock.
