"""Objective text/acoustic checks. Neither a MOS nor an emotion identity test."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import unicodedata

def normalized(text):
    from opencc import OpenCC
    import cn2an
    text = cn2an.transform(OpenCC("t2s").convert(text), "an2cn")
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower() if c.isalnum())

def edit_distance(a, b):
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j,y in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev = cur
    return prev[-1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--asr-model", type=Path, required=True)
    args = p.parse_args()
    from faster_whisper import WhisperModel
    import ctranslate2
    import librosa
    import numpy as np
    import soundfile as sf
    ctranslate2.set_random_seed(20261015)
    asr = WhisperModel(str(args.asr_model), device="cuda", compute_type="float16", cpu_threads=2, num_workers=1, local_files_only=True)
    metrics = json.loads(args.metrics.read_text())
    for item in metrics["samples"]:
        path = args.metrics.parent / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Raw audio hash changed")
        segments, _ = asr.transcribe(str(path), language="zh", beam_size=5, temperature=0,
            compression_ratio_threshold=None, log_prob_threshold=None, vad_filter=False, condition_on_previous_text=False)
        item["asr_text"] = "".join(s.text for s in segments)
        ref, hyp = normalized(item["text"]), normalized(item["asr_text"])
        item.update(writing_normalized_edits=edit_distance(ref,hyp), writing_normalized_characters=len(ref))
        item["writing_normalized_cer"] = item["writing_normalized_edits"] / len(ref)
        # Flag only; no manual removal/correction of recognizer errors.
        item["instruction_leak_suspected"] = any(t in item["asr_text"] for t in ["请非常", "用平静自然的语气", "惊讶的语气说一句话", "helpful assistant"])
        a,sr = sf.read(path)
        a16 = librosa.resample(a, orig_sr=sr, target_sr=16000)
        f0 = librosa.yin(a16, fmin=80, fmax=700, sr=16000, frame_length=1024, hop_length=256)
        rms = librosa.feature.rms(y=a16, frame_length=1024, hop_length=256)[0]
        f0 = f0[:len(rms)][rms[:len(f0)] > max(.003, float(rms.max())*.12)]
        item["acoustics"] = {"yin_f0_median_hz": float(np.median(f0)) if len(f0) else None,
            "yin_f0_iqr_hz": float(np.percentile(f0,75)-np.percentile(f0,25)) if len(f0) else None,
            "warning": "RMS-gated YIN proxy, not a validated emotion classifier or speaker similarity metric."}
        # Equal loudness listening copy; raw PCM untouched.
        gain = min(.9/max(float(np.abs(a).max()),1e-6), .1/max(float(np.sqrt(np.mean(a*a))),1e-6))
        listen = path.with_name(path.stem + "-listen.wav")
        sf.write(listen,a*gain,sr,subtype="PCM_16")
        item["listen_file"] = listen.name
        item["listen_gain"] = gain
        print("evaluated",item["id"],item["asr_text"],item["writing_normalized_cer"],flush=True)
    groups = {}
    for name in ["baseline","emotion","showcase","control"]:
        rows = [s for s in metrics["samples"] if s.get("group","baseline")==name]
        if not rows: continue
        chars=sum(x["writing_normalized_characters"] for x in rows)
        warm=rows[1:] if name=="baseline" else rows
        groups[name]={"utterances":len(rows),"writing_normalized_cer":sum(x["writing_normalized_edits"] for x in rows)/chars,
            "characters":chars,"warm_median_rtf":statistics.median(x["rtf"] for x in warm)}
    metrics["evaluation"]={"asr_snapshot":args.asr_model.name,"groups":groups,
        "method":"Whisper large-v3-turbo; beam 5, temperature 0, seed 20261015, no VAD/fallback/previous text. OpenCC t2s + cn2an an2cn + NFKC alphanumeric for both sides; no homophone fixes.",
        "emotion_recognition_validated":False,"human_listening_accepted":False}
    args.metrics.write_text(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
