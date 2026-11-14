# The notebooks that trained the three members

The three dense networks in `weights/` (v24, v27, v31) were each trained by one version of our research notebook on
Kaggle, `alok158/solar-filament`. The files here are those versions **as executed**, with their outputs: the
training logs, per-epoch validation scores and timings are part of each file. They were downloaded from each
version's Kaggle output (`__notebook__.ipynb`). The Kaggle notebook itself is made public when the competition
closes.

| File | Kaggle version | Completed | Training target | Augmentation | Run | Kept |
|---|---|---|---|---|---|---|
| `v24.ipynb` | 24 | 2026-10-10 | medoid annotator record | D4 + photometric | 23 of 35 epochs (8 h budget) | best fold-0 epoch (17) |
| `v27.ipynb` | 27 | 2026-10-13 | a random annotator record per sample | + rotation ±15°, scale ±12.5% | 26 of 26 epochs (8.7 h session) | best fold-0 epoch (21) |
| `v31.ipynb` | 31 | 2026-10-15 | medoid or random, 50/50 | as v27 | 30 of 30 epochs (one 9.9 h session) | final epoch |

**How to read them.** The notebook grew over the competition, so it still carries earlier, switched-off designs: a
YOLO detector and an embedding-based instance head, kept dormant behind run switches. The members were trained by the
*semantic* path (`RUN_SEMANTIC = True`, `SUBMIT_PIPELINE = 'semantic'`): the whole-disk ConvNeXt-Tiny U-Net with the
mask and spine heads described in the main notebook. Fold 0 (141 images) was held out in every version; the models
saw only folds 1-4.

**From these runs to `weights/`.**
1. Each version saved its model as `solar_filament_pq/semantic_models/fold_0/segmenter.pt` in its Kaggle output.
2. The three checkpoints were collected, unchanged, into a private Kaggle dataset (`v24.pt`, `v27.pt`, `v31.pt`).
3. `weights/<name>.fp16.pt` stores the same tensors in fp16. `MODELS.json` records each file's fingerprint, and the
   main notebook checks it every time the weights load.

**Checked on 2026-11-13.** Each `segmenter.pt` was downloaded again from its Kaggle version's output and
fingerprinted (sha256 over the tensors in fp16, `filament.model.weights_fingerprint`). All three match the shipped
files: v24 `d3a2e435f53e8dbe`, v27 `94c3a2a2d740398a`, v31 `d749aa06b75a627c` (`checkpoint_chain.json`).

**Same code as the package.** The research notebook embeds the research library that `filament/` was refactored
from. On the same inputs the two give identical input channels, network outputs, decoded instances, metric values,
augmentations, losses and filter fit (`provenance/check_release.py`, log in `provenance/check_release.log`). The member recipes are in
`filament/config.py` (`RECIPES`), and the main notebook can retrain any of them (`TRAINING = v24 | v27 | v31`).

The refiners are not here: they are trained by the main notebook itself (Section 7, `REFINER_TRAINING = full`).
