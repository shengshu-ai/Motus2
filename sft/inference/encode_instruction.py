#!/usr/bin/env python3
"""Precompute the UMT5 embedding of an instruction for /set_language uploads.

Usage:
  python inference/encode_instruction.py --wan-path /path/to/Wan2.2-TI2V-5B \
      --instruction "pick up the cup" --output cup.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "third_party"))

from wan.modules.t5 import T5EncoderModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wan-path", required=True, help="Wan2.2-TI2V-5B directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    wan_path = Path(args.wan_path).expanduser()
    encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=args.device,
        checkpoint_path=str(wan_path / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(wan_path / "google" / "umt5-xxl"),
    )
    with torch.no_grad():
        emb = encoder([args.instruction], args.device)[0]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb.detach().cpu(), out)
    print(f"saved {out} shape={tuple(emb.shape)}")


if __name__ == "__main__":
    main()
