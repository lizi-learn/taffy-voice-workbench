"""Same text, same voice reference; generated outputs and timings, not emotion-MOS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workbench import Request, IndexAdapter, CosyAdapter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["index", "cosy"], required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reference-text", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    import numpy as np
    import psutil
    import soundfile as sf
    import torch
    import nltk
    if os.environ.get("NLTK_DATA"):
        nltk.data.path[:] = [os.environ["NLTK_DATA"]]
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.4)
    torch.cuda.reset_peak_memory_stats()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model = (IndexAdapter if args.engine == "index" else CosyAdapter)(args.source, args.models)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    records = []
    for item in json.loads(args.prompts.read_text()):
        if args.engine not in item.get('engines', ['index','cosy']):
            print('skip_engine_specific_prompt', item['id'], flush=True)
            continue
        dest = args.out / f'{args.engine}-{item["id"]}-AI.wav'
        if dest.exists():
            raise FileExistsError(dest)
        req = Request(item["text"], args.reference, args.reference_text, dest,
            item.get("mood", "neutral"), item.get("strength", 1), item.get("speed", 1))
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.synthesize(req)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        a, sr = sf.read(dest)
        if not np.isfinite(a).all() or len(a) / sr < .2 or np.sqrt(np.mean(a*a)) < .0001:
            raise ValueError("Invalid or silent output")
        r = {**item, "engine": args.engine, "file": dest.name, "synthetic": True,
             "seconds": len(a)/sr, "compute_seconds": elapsed, "rtf": elapsed/(len(a)/sr),
             "peak_vram_bytes": torch.cuda.max_memory_allocated(), "rss_bytes": psutil.Process().memory_info().rss,
             "sample_rate": sr, "peak": float(np.abs(a).max()), "rms": float(np.sqrt(np.mean(a*a))),
             "clipped_fraction": float(np.mean(np.abs(a) >= .999)), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
        records.append(r)
        (args.out / f'{args.engine}-metrics.json').write_text(json.dumps({"load_seconds": load_seconds,
            "human_listening_accepted": False, "emotion_recognition_validated": False,
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "peak_safety_events": getattr(model, 'peak_safety_events', []),
            "notes": "4 CPU threads, batch 1, fp16/bf16, no TensorRT/DeepSpeed. PyTorch allocator memory excludes ONNX CUDA allocations. No MOS or target speaker identity claim.",
            "samples": records}, ensure_ascii=False, indent=2))
        print("sample_done", r, flush=True)

if __name__ == "__main__":
    main()
