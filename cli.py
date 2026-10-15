"""Local generation command; backend dependencies must be installed separately."""
import argparse
from pathlib import Path
from workbench import Request, MOODS, CosyAdapter, IndexAdapter

def main():
    p = argparse.ArgumentParser(description="Local AI-labelled TTS research. Use only permitted references.")
    p.add_argument("--engine", required=True, choices=["index", "cosy"])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reference-text", default="")
    p.add_argument("--text", required=True)
    p.add_argument("--mood", choices=MOODS, default="neutral")
    p.add_argument("--strength", type=float, default=1)
    p.add_argument("--speed", type=float, default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    req = Request(args.text, args.reference, args.reference_text, args.output, args.mood, args.strength, args.speed)
    req.validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = (IndexAdapter if args.engine == "index" else CosyAdapter)(args.source, args.models)
    model.synthesize(req)
    print("AI-generated output:", args.output)

if __name__ == "__main__":
    main()
