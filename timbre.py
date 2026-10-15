"""GPT-SoVITS and pinned public Bert-VITS2 adapters: one engine per process.

No synthetic emotion controls or pitch shifting. Install each upstream in its own
environment; only trusted, pinned source/model files belong in these paths.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import tempfile
from workbench import Request

@contextmanager
def source_directory(source):
    previous = Path.cwd()
    try:
        os.chdir(source)
        yield
    finally:
        os.chdir(previous)

def neutral_only(r):
    if r.mood != "neutral" or r.strength != 1:
        raise ValueError("This engine has reference/fixed prosody, not an independent emotion instruction")

@dataclass(frozen=True)
class GsvParameters:
    top_k: int = 15
    top_p: float = 1.
    temperature: float = 1.
    auxiliary_references: tuple[Path, ...] = ()

    def validate(self):
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or not 1 <= self.top_k <= 100:
            raise ValueError("top_k: integer 1–100")
        if not 0 < self.top_p <= 1 or not .1 <= self.temperature <= 1.5:
            raise ValueError("top_p: (0,1], temperature: .1–1.5")
        if len(self.auxiliary_references)>4 or not all(p.is_file() for p in self.auxiliary_references):
            raise ValueError("Provide at most four existing permitted auxiliary reference WAVs")

@dataclass(frozen=True)
class BertParameters:
    sdp_ratio: float = .2
    noise_scale: float = .6
    noise_scale_w: float = .8

    def validate(self):
        if not all(0 <= x <= 1 for x in (self.sdp_ratio,self.noise_scale,self.noise_scale_w)):
            raise ValueError("Bert stochastic parameters: 0–1")

class GsvAdapter:
    def __init__(self, source: Path, models: Path, gpt: Path, sovits: Path,
                 params=GsvParameters(), nltk_data: Path | None = None):
        params.validate()
        self.params = params
        self.source = source.resolve()
        models, gpt, sovits = models.resolve(), gpt.resolve(), sovits.resolve()
        if not gpt.is_file() or not sovits.is_file():
            raise FileNotFoundError("Provide explicit trusted GPT and SoVITS weights")
        os.environ["version"] = "v2ProPlus"
        sys.path[:0] = [str(self.source),str(self.source/"GPT_SoVITS")]
        if nltk_data:
            import nltk
            nltk.data.path[:] = [str(nltk_data.resolve())]
        self.config_dir = tempfile.TemporaryDirectory(prefix="taffy-gsv-")
        with source_directory(self.source):
            from TTS_infer_pack.TTS import TTS, TTS_Config
            config = TTS_Config({"custom": dict(device="cuda:0",is_half=True,version="v2ProPlus",
                t2s_weights_path=str(gpt),vits_weights_path=str(sovits),
                cnhuhbert_base_path=str(models/"chinese-hubert-base"),
                bert_base_path=str(models/"chinese-roberta-wwm-ext-large"))})
            config.configs_path = str(Path(self.config_dir.name)/"infer.yaml")
            self.model = TTS(config)

    def synthesize(self, r: Request):
        r.validate(); neutral_only(r)
        if not r.reference_text.strip():
            raise ValueError("GPT-SoVITS requires an accurate primary-reference transcript")
        import numpy as np
        import soundfile as sf
        params = dict(text=r.text,text_lang="zh",ref_audio_path=str(r.reference.resolve()),
            prompt_text=r.reference_text,prompt_lang="zh",
            aux_ref_audio_paths=[str(p.resolve()) for p in self.params.auxiliary_references],
            top_k=self.params.top_k,top_p=self.params.top_p,temperature=self.params.temperature,
            seed=r.seed,speed_factor=r.speed,batch_size=1,text_split_method="cut0",
            parallel_infer=True,split_bucket=False,return_fragment=False,streaming_mode=False,
            use_cuda_graph=False,use_flash_attention=False,super_sampling=False)
        with source_directory(self.source):
            chunks = list(self.model.run(params))
        if not chunks or len({sr for sr,_ in chunks})!=1:
            raise RuntimeError("Invalid engine output")
        sf.write(r.output,np.concatenate([a for _,a in chunks]),chunks[0][0],subtype="PCM_16")
        return str(r.output)

class BertAdapter:
    """Adapter for the pinned songdaooi Taffy Space's G_15800 architecture only."""
    def __init__(self, source: Path, config: Path, weights: Path,
                 params=BertParameters(), nltk_data: Path | None = None):
        params.validate()
        self.params, self.source = params, source.resolve()
        config, weights = config.resolve(), weights.resolve()
        sys.path.insert(0,str(self.source))
        if nltk_data:
            import nltk
            nltk.data.path[:] = [str(nltk_data.resolve())]
        import torch
        with source_directory(self.source):
            import utils
            from models import SynthesizerTrn
            from text.symbols import symbols
            self.hps = utils.get_hparams_from_file(str(config))
            h = self.hps
            self.model = SynthesizerTrn(len(symbols),h.data.filter_length//2+1,
                h.train.segment_size//h.data.hop_length,n_speakers=h.data.n_speakers,**h.model).cuda().eval()
            checkpoint = torch.load(weights,map_location="cpu",weights_only=True)
            compatible = self.model.load_state_dict(checkpoint["model"],strict=False)
            if compatible.missing_keys or not all(".cond_pre." in k or ".cond_layer." in k for k in compatible.unexpected_keys):
                raise RuntimeError(f"Unexpected checkpoint architecture: {compatible}")

    def synthesize(self, r: Request):
        r.validate(require_reference=False); neutral_only(r)
        import random
        import numpy as np
        import soundfile as sf
        import torch
        random.seed(r.seed); np.random.seed(r.seed); torch.manual_seed(r.seed)
        with source_directory(self.source):
            import commons
            from text import cleaned_text_to_sequence, get_bert
            from text.cleaner import clean_text
            text,phones,tones,word2ph = clean_text(r.text,"ZH")
            phones,tones,languages = cleaned_text_to_sequence(phones,tones,"ZH")
            if self.hps.data.add_blank:
                phones,tones,languages = [commons.intersperse(x,0) for x in (phones,tones,languages)]
                word2ph = [x*2 for x in word2ph]; word2ph[0]+=1
            bert = get_bert(text,word2ph,"ZH").cuda().unsqueeze(0)
            with torch.inference_mode():
                audio = self.model.infer(torch.tensor([phones],device="cuda"),
                    torch.tensor([len(phones)],device="cuda"),torch.tensor([0],device="cuda"),
                    torch.tensor([tones],device="cuda"),torch.tensor([languages],device="cuda"),bert,
                    sdp_ratio=self.params.sdp_ratio,noise_scale=self.params.noise_scale,
                    noise_scale_w=self.params.noise_scale_w,length_scale=1/r.speed)[0][0,0].float().cpu().numpy()
        sf.write(r.output,audio,self.hps.data.sampling_rate,subtype="PCM_16")
        return str(r.output)
