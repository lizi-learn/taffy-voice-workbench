"""Small, local-only adapter API. No GPU ports, accounts or reference audio included."""
from dataclasses import dataclass
from pathlib import Path
import sys

MOODS = ("neutral", "happy", "sad", "angry", "surprised", "gentle")
VECTORS = {
    "neutral": [0, 0, 0, 0, 0, 0, 0, .6],
    "happy": [.7, 0, 0, 0, 0, 0, 0, 0],
    "sad": [0, 0, .7, 0, 0, 0, 0, 0],
    "angry": [0, .7, 0, 0, 0, 0, 0, 0],
    "surprised": [0, 0, 0, 0, 0, 0, .7, 0],
    "gentle": [0, 0, 0, 0, 0, 0, 0, .35],
}
INSTRUCTIONS = {
    "neutral": "请用平静自然的语气说一句话。",
    "happy": "请非常开心地说一句话。",
    "sad": "请非常伤心地说一句话。",
    "angry": "请非常生气地说一句话。",
    "surprised": "请用惊讶的语气说一句话。",
    "gentle": "请用温柔轻柔的语气说一句话。",
}

@dataclass(frozen=True)
class Request:
    text: str
    reference: Path | None
    reference_text: str
    output: Path
    mood: str = "neutral"
    strength: float = 1.0
    speed: float = 1.0
    seed: int = 20261015

    def validate(self, require_reference=True):
        if not 1 <= len(self.text.strip()) <= 300:
            raise ValueError("Text must have 1–300 characters")
        if self.mood not in MOODS:
            raise ValueError("Unsupported mood")
        if not 0 <= self.strength <= 1 or not .8 <= self.speed <= 1.25:
            raise ValueError("strength: 0–1, speed: .8–1.25")
        if require_reference and (self.reference is None or not self.reference.is_file()):
            raise ValueError("Provide a permitted local reference WAV")
        if self.output.exists():
            raise FileExistsError("Refusing to overwrite existing output")

class IndexAdapter:
    def __init__(self, source: Path, models: Path, peak_safe=True):
        sys.path.insert(0, str(source.resolve()))
        from indextts.infer_v2_5 import IndexTTS2
        self.model = IndexTTS2(cfg_path=str(models / "config.yaml"), model_dir=str(models),
            use_bf16=True, use_cuda_kernel=False, use_deepspeed=False, use_qwen_emo=False)
        self.peak_safety_events = []
        if peak_safe:
            # Keep headroom BEFORE upstream float-to-PCM clipping. No weight or timbre edits.
            def headroom(_module, _inputs, output):
                peak = float(output.detach().abs().max())
                if peak > .95:
                    self.peak_safety_events.append({"input_peak": peak, "gain": .95/peak})
                    return output * (.95/peak)
                return output
            self.model.bigvgan.register_forward_hook(headroom)

    def synthesize(self, r: Request):
        r.validate()
        import torch
        torch.manual_seed(r.seed)
        return self.model.infer(spk_audio_prompt=str(r.reference), text=r.text, lang="ZH",
            output_path=str(r.output), emo_vector=[v * r.strength for v in VECTORS[r.mood]],
            use_random=False, duration_factor=1 / r.speed, verbose=False)

class CosyAdapter:
    def __init__(self, source: Path, models: Path):
        sys.path[:0] = [str(source.resolve()), str((source / "third_party/Matcha-TTS").resolve())]
        from cosyvoice.cli.cosyvoice import AutoModel
        self.model = AutoModel(model_dir=str(models), load_trt=False, fp16=True)

    def synthesize(self, r: Request):
        r.validate()
        if r.strength != 1:
            raise ValueError("CosyVoice instruct has no calibrated intensity slider; use strength=1")
        import numpy as np
        import torch
        import soundfile as sf
        import random
        random.seed(r.seed)
        np.random.seed(r.seed)
        torch.manual_seed(r.seed)
        # Emotion belongs to the control prompt, never prepended to spoken text.
        prompt = "You are a helpful assistant. " + INSTRUCTIONS[r.mood] + "<|endofprompt|>"
        chunks = [x["tts_speech"].detach().cpu().float() for x in self.model.inference_instruct2(
            r.text, prompt, str(r.reference), stream=False, speed=r.speed, text_frontend=False)]
        if not chunks:
            raise RuntimeError("Engine returned no audio")
        audio = torch.cat(chunks, dim=1).squeeze(0).numpy()
        sf.write(r.output, audio, self.model.sample_rate, subtype="PCM_16")
        return str(r.output)
