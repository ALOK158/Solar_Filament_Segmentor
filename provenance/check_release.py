"""Equivalence of the clean release package (release/filament) with the libraries that produced the submitted
results (semantic_filament_lib v39 + solar_filament_lib + blend_core), on CPU.

1. preprocessing channels on 3 images: identical arrays
2. model forward (v24 fp32 weights, 256 px input): identical logits; fp16-stored weights: max |dprob|
3. decode on all 141 shipped fold-0 maps: identical instance lists
4. filter fit on the shipped maps: same table, threshold and PQ as final/local_filter_report_nm.json
5. RLE and PQ on one image: identical
6. loss + augmentation on a fixed rng: identical
Usage: python final/check_release.py   (from the SPACE root)  -> final/check_release.log
"""
import json, sys, time
from dataclasses import replace
from pathlib import Path

SPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SPACE / 'release'))
sys.path.insert(0, str(SPACE / 'blend' / 'dataset_flat'))
sys.path.insert(0, str(SPACE))
import numpy as np, pandas as pd, torch
import solar_filament_lib as base
import semantic_filament_lib as sfl
import blend_core as bc
from filament import config as C, data, preprocess, model as M, inference, decode, instance_filter as F, metrics, submission, maps as mp, losses, augment

LOG = open(SPACE / 'final' / 'check_release.log', 'w', encoding='utf-8')
def log(m):
    print(m, flush=True); LOG.write(m + '\n'); LOG.flush()

DATA = SPACE / 'data'
old_idx = base.build_coco_indices(base.load_coco(DATA / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
new_idx = data.load_annotations(DATA / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
files = sorted(p.name for p in (DATA / 'train_images').iterdir())[:3]

# 1. channels
for f in files:
    img = sfl.read_gray(DATA / 'train_images' / f)
    a = sfl.to_channels(sfl.resize_to_input(img, 1536), recipe='bgnorm_clahe_ridge')
    b = preprocess.network_input(img)
    assert a.dtype == b.dtype and np.array_equal(a, b), f
log('1. channels identical on 3 images')

# 2. forward
cfg_old = replace(sfl.SemanticCFG(), use_tta=True, tta_transpose=False)
old_model = sfl.load_segmenters([SPACE / 'blend' / 'dataset_flat' / 'v24.pt'], cfg_old, torch.device('cpu'))[0]
new_model = M.build_model()
state = torch.load(SPACE / 'blend' / 'dataset_flat' / 'v24.pt', map_location='cpu')
new_model.load_state_dict(state); new_model.eval()
x = torch.from_numpy(preprocess.network_input(sfl.read_gray(DATA / 'train_images' / files[0]), C.InputConfig(input_size=256)))[None]
with torch.no_grad():
    lo, ln = old_model(x), new_model(x)
assert torch.equal(lo, ln), (lo - ln).abs().max()
fp = M.save_member(state, SPACE / 'scratchpad' / 'v24_check.fp16.pt')
half = M.load_member(SPACE / 'scratchpad' / 'v24_check.fp16.pt')
with torch.no_grad():
    lh = half(x)
dp = (lo.sigmoid() - lh.sigmoid()).abs()
log(f'2. forward identical (256 px); fp16-stored weights: max |dprob| {dp.max():.2e}, mean {dp.mean():.2e}; '
    f'params {M.count_parameters(new_model)}; fingerprint {fp}')
del old_model, new_model, half

# 3. decode on shipped maps
store = mp.MapStore(SPACE / 'blend' / 'dataset_flat' / 'fold0_blend_b24_27_31.npz')
pin = json.loads((SPACE / 'blend' / 'dataset_flat' / 'decode_pin.json').read_text())['decode']
cfg_nm = replace(sfl.SemanticCFG(), **{**pin, 'max_gap_px': 0.0}, use_tta=True, tta_transpose=False)
t0, n_inst = time.time(), 0
for f in store:
    p, s = store[f]
    a = sfl.decode_instances(p, cfg_nm)
    b = decode.decode_instances(p)
    assert len(a) == len(b) and all(np.array_equal(u, v) for u, v in zip(a, b)), f
    n_inst += len(b)
log(f'3. decode identical on {len(store)} maps ({n_inst} instances, {time.time() - t0:.0f}s)')

# 5. RLE + PQ on one image
f0 = next(iter(store))
inst = decode.decode_instances(store[f0][0])
assert [submission.encode_rle(m) for m in inst] == [base.encode_submission_mask(m) for m in inst]
ro = pd.DataFrame(base.evaluate_prediction_for_file(f0, inst, old_idx, cfg_nm))
rn = pd.DataFrame(metrics.score_image(f0, inst, new_idx))
assert np.allclose(ro[['pq', 'tp', 'fp', 'fn', 'sum_iou']].to_numpy(float), rn[['pq', 'tp', 'fp', 'fn', 'sum_iou']].to_numpy(float))
assert base.choose_medoid_annotators(old_idx) == data.medoid_records(new_idx)
log('5. RLE, per-record PQ and medoid choice identical')

# 6. loss + augmentation
tc = C.RECIPES['v31']
old_cfg = replace(sfl.SemanticCFG(), random_annotator_prob=0.5, photometric_v2_prob=0.3)
img = sfl.resize_to_input(sfl.read_gray(DATA / 'train_images' / files[0]), 512)
tgt = np.stack([np.zeros((512, 512), np.uint8)] * 2, -1); tgt[200:220, 100:300] = 1
ia, ca, ta = sfl.augment(img, tgt, np.random.default_rng(7), old_cfg)
ib, cb, tb = augment.augment(img, tgt, np.random.default_rng(7), replace(tc, photometric_v2_prob=0.3))
assert np.array_equal(ia, ib) and ca == cb and np.array_equal(ta, tb)
torch.manual_seed(0)
logits = torch.randn(2, 2, 64, 64)
target = (torch.rand(2, 1, 64, 64) > 0.8).float(); spine = (torch.rand(2, 1, 64, 64) > 0.9).float()
import cv2
comps = torch.stack([torch.from_numpy(cv2.connectedComponents(t[0].numpy().astype(np.uint8), connectivity=8)[1].astype(np.int32))[None] for t in target])
for pseudo in (False, True):
    lo_ = sfl.segmentation_loss(logits, target, sfl.SemanticCFG(), spine, comps, pseudo=pseudo)
    ln_ = losses.segmentation_loss(logits, target, spine, comps, tc, pseudo=pseudo)
    assert torch.equal(lo_, ln_), (pseudo, lo_, ln_)
log('6. augmentation (affine + photometric_v2) and loss (labelled and pseudo) identical')

# 4. filter fit
t0 = time.time()
filt = F.fit_filter(store, new_idx, log=log)
expect = json.loads((SPACE / 'final' / 'local_filter_report_nm.json').read_text())
for k in ('instances', 'unfiltered_pq', 'filtered_pq', 'threshold', 'deploy_threshold'):
    log(f'   {k:16s} new {filt.report[k]:.6f} | old {expect[k]:.6f}')
    assert abs(float(filt.report[k]) - float(expect[k])) < 1e-9, k
filt.save(SPACE / 'scratchpad' / 'filter_check.pkl')
log(f'4. filter fit identical ({time.time() - t0:.0f}s); AUC(any annotator) {filt.report["auc_matched_any"]:.3f}')
log('ALL CHECKS PASS')
