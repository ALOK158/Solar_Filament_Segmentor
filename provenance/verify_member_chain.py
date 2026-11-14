"""Provenance check (needs Kaggle API credentials; run from the repository root): each member's checkpoint, downloaded from the Kaggle version that trained it
(alok158/solar-filament v24 / v27 / v31, solar_filament_pq/semantic_models/fold_0/segmenter.pt), must have the same
fp16 fingerprint as the shipped weights/<name>.fp16.pt (MODELS.json). One checkpoint in memory at a time."""
import gc, json, sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent          # the repository root
sys.path.insert(0, str(REPO))
import torch
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglesdk.kernels.types.kernels_api_service as T
from filament import model as M

cards = json.loads((REPO / 'weights' / 'MODELS.json').read_text())
api = KaggleApi(); api.authenticate()
results = {}
with api.build_kaggle_client() as k:
    for name, version in (('v24', 24), ('v27', 27), ('v31', 31)):
        r = T.ApiDownloadKernelOutputRequest()
        r.owner_slug, r.kernel_slug, r.version_number = 'alok158', 'solar-filament', version
        r.file_path = 'solar_filament_pq/semantic_models/fold_0/segmenter.pt'
        path = Path(tempfile.gettempdir()) / f'segmenter_v{version}.pt'
        import requests
        response = k.kernels.kernels_api_client.download_kernel_output(r)    # lazy body: only the final URL is used
        url, total = response.url, int(response.headers.get('Content-Length', 0))
        response.close()
        with open(path, 'wb') as handle:
            done, failures = 0, 0
            while done < total and failures < 40:          # the connection drops every few MB: resume by range
                try:
                    with requests.get(url, headers={'Range': f'bytes={done}-'}, stream=True, timeout=60) as part:
                        for chunk in part.iter_content(1 << 20):
                            handle.write(chunk); done += len(chunk)
                except Exception:
                    failures += 1
            if done != total:
                raise RuntimeError(f'{name}: got {done} of {total} bytes')
        print(f'{name}: downloaded {done / 1e6:.0f} MB ({failures} resumes)', flush=True)
        blob = torch.load(path, map_location='cpu', weights_only=False)
        state = blob.get('state_dict', blob.get('model', blob)) if isinstance(blob, dict) else blob
        keys = list(blob.keys())[:6] if isinstance(blob, dict) else type(blob).__name__
        fp = M.weights_fingerprint(M._stored_tensors(state))
        ok = fp == cards['members'][name]['fingerprint']
        results[name] = {'version': version, 'file_MB': round(path.stat().st_size / 1e6, 1), 'top_keys': keys,
                         'fingerprint': fp, 'matches_shipped': ok}
        print(name, results[name], flush=True)
        path.unlink(); del blob, state; gc.collect()
(REPO / 'training_notebooks' / 'checkpoint_chain.json').write_text(json.dumps(results, indent=1, default=str))
