"""Pin official Hub metadata, then create a hash-verifiable download plan.

Weights and voice recordings are deliberately excluded from this repository.
"""
import argparse
import json
from pathlib import Path
import requests

SPECS = [
    ("IndexTeam/IndexTTS-2.5", "IndexTeam/IndexTTS-2.5", "index", lambda n: not n.startswith("qwen0.6bemo4-merge/")),
    ("facebook/w2v-bert-2.0", "AI-ModelScope/w2v-bert-2.0", "index/hf_cache/w2v-bert-2.0", lambda n: n in {"config.json", "model.safetensors", "preprocessor_config.json"}),
    ("amphion/MaskGCT", "amphion/MaskGCT", "index/hf_cache/semantic_codec", lambda n: n == "semantic_codec/model.safetensors"),
    ("funasr/campplus", "iic/speech_campplus_sv_zh-cn_16k-common", "index/hf_cache", lambda n: n == "campplus_cn_common.bin"),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "nvidia/bigvgan_v2_22khz_80band_256x", "index/hf_cache/bigvgan", lambda n: n in {"config.json", "bigvgan_generator.pt", "LICENSE"}),
    ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "cosy", lambda n: n.startswith("CosyVoice-BlankEN/") or n in {"cosyvoice3.yaml", "config.json", "configuration.json", "campplus.onnx", "flow.pt", "hift.pt", "llm.pt", "speech_tokenizer_v3.onnx", "README.md", "LICENSE"}),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    files, repos = [], []
    for repo, mirror, directory, accept in SPECS:
        r = requests.get(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=60)
        r.raise_for_status()
        meta = r.json()
        repos.append({"repo": repo, "revision": meta["sha"], "license": meta.get("cardData", {}).get("license", "check upstream")})
        for f in meta["siblings"]:
            n = f["rfilename"]
            if not accept(n) or n == ".gitattributes":
                continue
            dest = n.removeprefix("semantic_codec/") if repo == "amphion/MaskGCT" else n
            files.append({"repo": repo, "revision": meta["sha"], "source": n,
                          "mirror_repo": mirror, "destination": f"{directory}/{dest}",
                          "bytes": f.get("size"), "sha256": f.get("lfs", {}).get("sha256")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"repositories": repos, "files": files}, ensure_ascii=False, indent=2))
    print(json.dumps({"files": len(files), "bytes": sum(x["bytes"] or 0 for x in files), "repositories": repos}))

if __name__ == "__main__":
    main()
