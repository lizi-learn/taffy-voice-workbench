# SPDX-License-Identifier: MIT
"""Same weights/reference/seed; instruction vs transcript-conditioned Cosy3.

No listening preference is inferred from ASR, pitch or compute measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workbench import CosyAdapter, Request

def main():
    p = argparse.ArgumentParser()
    for name in ['source', 'models', 'reference', 'prompts', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--reference-text', required=True)
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError('Use a fresh output directory')
    a.out.mkdir(parents=True, mode=0o700)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4')
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    start = time.perf_counter()
    model = CosyAdapter(a.source, a.models)
    torch.cuda.synchronize()
    load = time.perf_counter() - start
    prompts = json.loads(a.prompts.read_text())
    if isinstance(prompts, dict): prompts = prompts['prompts']
    for mode in ['instruct2', 'zero_shot']:
        model.mode = mode
        folder = a.out / mode
        folder.mkdir()
        records = []
        torch.cuda.reset_peak_memory_stats()
        for item in prompts:
            path = folder / f'cosy-{mode}-{item["id"]}-AI.wav'
            req = Request(item['text'], a.reference, a.reference_text, path)
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.synthesize(req)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            audio, sr = sf.read(path)
            if not np.isfinite(audio).all() or len(audio)/sr < .2:
                raise ValueError('Invalid output')
            row = dict(**item, file=path.name, seconds=len(audio)/sr,
                compute_seconds=elapsed, rtf=elapsed/(len(audio)/sr), sample_rate=sr,
                peak_vram_bytes=torch.cuda.max_memory_allocated(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                clipped_fraction=float(np.mean(np.abs(audio) >= .999)),
                conditioning=mode, seed=req.seed, synthetic=True)
            row.setdefault('group','baseline')
            records.append(row)
            report = dict(mode=mode, load_seconds=load,
                reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
                human_listening_accepted=False, samples=records,
                method='Same live model, reference WAV and accurate transcript, seed reset per prompt. Instruct2 first; excludes first prompt for warm timing. PyTorch allocator memory excludes ONNX CUDA allocations. No timbre score.')
            (folder/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print('mode_sample', mode, row['id'], round(row['rtf'], 3), flush=True)

if __name__ == '__main__': main()
