# State Tracker

| Loop ID | Baseline Metrics Table | Iteration Log (Loop ID, Hypothesis, Result, Action) |
|---------|------------------------|---------------------------------------------------|
| 0       | PCK@0.5: TBD, MPJPE: TBD | N/A                                               |
| 1       | PCK@0.5: 74.2%, MPJPE: 27.8 | 1, Thermal Diffusion, FAILURE, Discovered rotation bug in augmentations.py; metrics were invalid due to coordinate flipping. |
| 2       | PCK@0.5: 73.0%, MPJPE: 29.7 | 2, loop2_fixed_aug, SUCCESS, Fixed rotation bug and pathing. This is the new stable baseline. |
| 3       | PCK@0.5: 74.9%, MPJPE: 27.4 | 3, loop3_improved_thermal_full_data, SUCCESS, Improved Thermal Diffusion (wavy edges, full cover) + Full Dataset (80 subjects). Clear improvement over baseline. |
