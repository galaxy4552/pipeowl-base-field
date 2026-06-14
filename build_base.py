# build_base.py
# Build embeddings using BAAI/bge-m3
# Vocab source: base_vocab.json (list[str])
#
# Usage (cmd):
# python build_base.py --out_dir "C:\code\base"
# 如果 vocab 很大（例如 16萬詞）
# python build_base.py --out_dir "C:\code\base" --batch 32

import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_VOCAB = r"C:\code\base\data\base_vocab.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--save_dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    model_path = args.model
    vocab_path = Path(args.vocab)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npy = out_dir / "base.npy"

    # -------------------------
    # load vocab
    # -------------------------
    if not vocab_path.exists():
        raise FileNotFoundError(vocab_path)

    with open(args.vocab, "r", encoding="utf-8-sig") as f:
        vocab_list = json.load(f)
        
    # 如果是 [[...]] 這種結構，展平
    if len(vocab_list) == 1 and isinstance(vocab_list[0], list):
        vocab_list = vocab_list[0]

    # 確保全部是字串
    vocab_list = [str(x) for x in vocab_list if x is not None]

    # -------------------------
    # load model
    # -------------------------
    print("[INFO] loading model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        use_safetensors=True
    )

    device = torch.device(args.device)
    model.eval()
    model.to(device)

    # -------------------------
    # encode
    # -------------------------

    total = len(vocab_list)
    bs = int(args.batch)
    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32

    print("[INFO] encoding vocab...")
    print("[INFO] total tokens =", total)
    print("[INFO] device =", device)
    print("[INFO] save dtype =", args.save_dtype)

    # BGE-M3 hidden size is available from model config; no dummy forward needed.
    dim = int(model.config.hidden_size)

    vecs = np.zeros((total, dim), dtype=save_dtype)

    idx = 0

    for i in tqdm(range(0, total, bs)):
        batch = vocab_list[i:i + bs]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)

        vecs[idx:idx+len(batch)] = embeddings.detach().cpu().numpy().astype(save_dtype, copy=False)
        idx += len(batch)

    # -------------------------
    # save
    # -------------------------
    np.save(out_npy, vecs)
    print(f"[INFO] saved {args.save_dtype}")

    print("DONE")
    print("embeddings:", out_npy)


if __name__ == "__main__":
    main()
