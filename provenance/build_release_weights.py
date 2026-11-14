"""fp32 member checkpoints -> release/weights/<name>.fp16.pt + release/weights/MODELS.json (model cards with the
weights fingerprints). Also copies the small artifacts the notebook reads.
Usage: python final/build_release_weights.py   (from the SPACE root)"""
import json, shutil, sys
from pathlib import Path

SPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SPACE / 'release'))
import torch
from filament import model as M

SRC = SPACE / 'blend' / 'dataset_flat'
OUT = SPACE / 'release' / 'weights'
COMMON = {
    'architecture': 'FilamentUNet: ConvNeXt-Tiny encoder (timm convnext_tiny.dinov3_lvd1689m, DINOv3-distilled init) + '
                    'U-Net decoder (256/128/64/32) + full-resolution stem; heads: filament mask, filament spine',
    'parameters': 31.9e6,
    'input': '1536 x 1536, 3 channels (background-normalised, CLAHE, dark-ridge response)',
    'training_data': 'MAGFiLO folds 1-4 (566 images); fold 0 (141 images) held out for validation',
    'optimiser': 'AdamW lr 3e-4 (encoder x0.1), weight decay 1e-4, batch 1 x accumulation 2, cosine to 0, AMP with fp32 encoder',
    'loss': 'BCE + 0.5 per-instance Dice + 0.5 image Dice + 0.5 Skeleton Recall (drawn spine) + 0.3 spine-head BCE',
    'hardware': 'one NVIDIA Tesla T4 (Kaggle), 13-17 min per epoch',
}
CARDS = {
    'v24': {'targets': 'medoid annotator (the record agreeing most with the others)',
            'augmentation': 'D4 rotations/flips, gamma, blur, noise, CLAHE strength',
            'schedule': '35 epochs requested; stopped by the session budget after 23; best validated epoch (17) kept',
            'fold0_pq_filtered': 0.4494, 'public_lb': 0.39},
    'v27': {'targets': 'a random annotator record per sample (label augmentation)',
            'augmentation': 'v24 + rotation +-15 deg and scale +-12.5% on half of the samples',
            'schedule': '26 epochs, best validated epoch (24) kept',
            'fold0_pq_filtered': 0.4427, 'public_lb': 0.38},
    'v31': {'targets': '50/50 medoid / random annotator per sample',
            'augmentation': 'as v27',
            'schedule': '30 epochs, final epoch kept (no selection on validation images)',
            'fold0_pq_filtered': 0.4375, 'public_lb': 0.38},
}
ENSEMBLE = {'members': ['v24', 'v27', 'v31'], 'combination': 'mean of sigmoid probabilities over 3 members x 4 flips',
            'fold0_pq_filtered_pinned_decode': 0.4529, 'fold0_pq_filtered_final_decode': 0.4567,
            'fold0_pq_split_averaged_final_decode': 0.4548, 'public_lb': 0.39,
            'note': 'single members score 0.437-0.449 with their own filters; the ensemble is the only change that '
                    'cleared the +0.005 adoption bar'}

OUT.mkdir(parents=True, exist_ok=True)
cards = {'common': COMMON, 'ensemble': ENSEMBLE, 'members': {}}
for name, card in CARDS.items():
    state = torch.load(SRC / f'{name}.pt', map_location='cpu')
    fingerprint = M.save_member(state, OUT / f'{name}.fp16.pt', {'name': name, **card})
    size = (OUT / f'{name}.fp16.pt').stat().st_size / 2 ** 20
    cards['members'][name] = {**card, 'file': f'weights/{name}.fp16.pt', 'megabytes': round(size, 1),
                              'fingerprint': fingerprint}
    print(name, fingerprint, f'{size:.1f} MB')
    del state
(OUT / 'MODELS.json').write_text(json.dumps(cards, indent=1), encoding='utf-8')
ART = SPACE / 'release' / 'artifacts'
shutil.copy2(SPACE / 'scratchpad' / 'v31out' / 'solar_filament_pq' / 'fold_manifest.csv', ART / 'fold_manifest.csv')
shutil.copy2(SPACE / 'scratchpad' / 'onnxout' / 'onnx_benchmark.json', ART / 'onnx_benchmark.json')
print('wrote', OUT / 'MODELS.json')
