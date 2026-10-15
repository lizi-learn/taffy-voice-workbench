"""Local generation command; backend dependencies must be installed separately."""
import argparse
from pathlib import Path
from workbench import Request, MOODS, CosyAdapter, IndexAdapter
from timbre import GsvAdapter, BertAdapter, GsvParameters, BertParameters, neutral_only

def main():
    p = argparse.ArgumentParser(description="Local AI-labelled TTS research. Use only permitted references.")
    p.add_argument("--engine", required=True, choices=["gsv", "bert", "index", "cosy"])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--models", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--gpt", type=Path)
    p.add_argument("--sovits", type=Path)
    p.add_argument("--bert-config", type=Path)
    p.add_argument("--bert-weights", type=Path)
    p.add_argument("--nltk-data", type=Path)
    p.add_argument("--aux-reference", type=Path, action="append", default=[])
    p.add_argument("--temperature", type=float, default=1)
    p.add_argument("--top-p", type=float, default=1)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--noise-scale", type=float, default=.6)
    p.add_argument("--noise-scale-w", type=float, default=.8)
    p.add_argument("--reference-text", default="")
    p.add_argument("--text", required=True)
    p.add_argument("--mood", choices=MOODS, default="neutral")
    p.add_argument("--strength", type=float, default=1)
    p.add_argument("--speed", type=float, default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    req = Request(args.text, args.reference, args.reference_text, args.output, args.mood, args.strength, args.speed)
    req.validate(require_reference=args.engine!="bert")
    if args.engine in {"gsv","bert"}:
        neutral_only(req)
    if args.engine!="bert" and args.models is None:
        p.error("--models is required for this engine")
    if args.engine=="gsv" and (args.gpt is None or args.sovits is None or not args.reference_text.strip()):
        p.error("GPT-SoVITS requires --gpt, --sovits and --reference-text")
    if args.engine=="bert" and (args.bert_config is None or args.bert_weights is None):
        p.error("Bert requires --bert-config and --bert-weights")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.engine=="gsv":
        model=GsvAdapter(args.source,args.models,args.gpt,args.sovits,
            GsvParameters(args.top_k,args.top_p,args.temperature,tuple(args.aux_reference)),args.nltk_data)
    elif args.engine=="bert":
        model=BertAdapter(args.source,args.bert_config,args.bert_weights,
            BertParameters(noise_scale=args.noise_scale,noise_scale_w=args.noise_scale_w),args.nltk_data)
    else:
        model = (IndexAdapter if args.engine == "index" else CosyAdapter)(args.source, args.models)
    model.synthesize(req)
    print("AI-generated output:", args.output)

if __name__ == "__main__":
    main()
