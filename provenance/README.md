# Provenance checks

The scripts that tie this repository to the runs that produced its weights. They are evidence, not part of the
pipeline: the notebook never needs them.

| File | What it checks | Result |
|---|---|---|
| `verify_member_chain.py` | Downloads each member's checkpoint from the output of the Kaggle notebook version that trained it (`alok158/solar-filament` v24, v27, v31) and compares its fp16 fingerprint with `weights/MODELS.json`. Needs Kaggle API credentials; runs from the repository root. | all three match: `training_notebooks/checkpoint_chain.json` |
| `build_release_weights.py` | Converted the three checkpoints to the fp16 files in `weights/` and wrote their fingerprints. | `weights/MODELS.json` |
| `check_release.py` | Runs the `filament/` package and the research code it was refactored from on the same inputs: input channels, network outputs, fp16 rounding, decoded instances on all 141 validation maps, RLE, PQ, medoid choice, augmentation, losses and the filter fit. | `check_release.log`: all checks pass; fp16 weights move probabilities by at most 1.6e-6 |

`build_release_weights.py` and `check_release.py` ran in our working folder, where this repository was the `release/`
subfolder, so their paths start one level up. `check_release.py` also imports the research library
(`solar_filament_lib`, `semantic_filament_lib`, `blend_core`); that library is embedded in the executed notebooks in
`training_notebooks/`.

The refiner weights need no separate chain: the notebook trains them itself (Section 7, `FILAMENT_REFINER_TRAINING=full`), and
its executed copy in this repository is that run.
