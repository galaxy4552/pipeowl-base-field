import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ========================
# 路徑設定（依你的環境）
# ========================
DEFAULT_MODEL = "BAAI/bge-m3"

BASE_JSON = r"C:\code\base\data\tokenizer.json"
# 如果你有這個檔，會走「模式 A」直接讀向量（快很多）
BASE_NPY  = r"C:\code\base\data\base.npy"

OUT_NPY   = r"C:\code\base\data\field_prior.npy"

# 你要的「delta方向」：把詞推向「解釋/定義」語境
def expand_word(w: str) -> str:
    return f"{w} 是什麼意思？"

# ========================
# 0. 讀取詞表
# ========================
print("[FieldPrior] loading vocab json...")
with open(BASE_JSON, "r", encoding="utf-8-sig") as f:
    vocab = json.load(f)

print("[FieldPrior] vocab size =", len(vocab))
assert isinstance(vocab, list) and len(vocab) > 0

# ========================
# 1. 取得 base_vectors（優先讀 npy）
# ========================
base_vectors = None
base_npy_path = Path(BASE_NPY)

if base_npy_path.exists():
    print("[FieldPrior] loading base vectors from NPY (fast path)...")
    base_vectors = np.load(BASE_NPY).astype("float32")
    print("[FieldPrior] base_vectors shape =", base_vectors.shape)
else:
    print("[FieldPrior] base NPY not found, encoding base vectors from model...")
    encoder = SentenceTransformer(DEFAULT_MODEL)
    batch_size = 256
    vecs = []
    for i in tqdm(range(0, len(vocab), batch_size)):
        batch = vocab[i:i+batch_size]
        emb = encoder.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            show_progress_bar=False
        )
        vecs.append(emb.astype("float32"))
    base_vectors = np.vstack(vecs)
    print("[FieldPrior] base_vectors shape =", base_vectors.shape)

# 防呆：base_vectors 行數要等於 vocab
assert base_vectors.shape[0] == len(vocab), "base_vectors rows != vocab size"

# ========================
# 2. 建立 field_vec（平均「解釋化位移」）
#    field_vec = mean( E(expand(w)) - base_vectors[w] )
#
# 注意：這裡直接使用 base_vectors 作為 E(w)，避免 base.npy 與
# SentenceTransformer 重新 encode 出來的 pooling 方法不一致。
# ========================
print("[FieldPrior] loading model for field prior...")
encoder = SentenceTransformer(DEFAULT_MODEL)

batch_size = 256
deltas = []

print("[FieldPrior] computing field_vec (mean expanded - base_vectors)...")
for i in tqdm(range(0, len(vocab), batch_size)):
    batch = vocab[i:i+batch_size]

    # 使用已建立好的 base_vectors，確保 field prior 與 base.npy 在同一個 embedding space。
    base_emb = base_vectors[i:i+len(batch)]

    exp_batch = [expand_word(w) for w in batch]
    exp_emb = encoder.encode(
        exp_batch,
        batch_size=len(exp_batch),
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    deltas.append((exp_emb - base_emb).astype("float32"))

deltas = np.vstack(deltas)
field_vec = deltas.mean(axis=0).astype("float32")

# normalize field_vec，讓 beta 比較好調
norm = np.linalg.norm(field_vec)
if norm > 0:
    field_vec /= norm

print("[FieldPrior] field_vec norm =", float(np.linalg.norm(field_vec)))

# ========================
# 3. 投影到每個詞 → field_prior
#    field_prior[i] = dot(base_vectors[i], field_vec)
# ========================
print("[FieldPrior] projecting field_vec onto vocab vectors...")
field_prior = (base_vectors @ field_vec).astype("float32")

# （可選）中心化：讓它像 bias 一樣「只拉相對差異」
field_prior -= float(field_prior.mean())

np.save(OUT_NPY, field_prior)
print("[FieldPrior] saved:", OUT_NPY)
print("[FieldPrior] field_prior shape:", field_prior.shape)
print("[FieldPrior] stats: min/mean/max =",
      float(field_prior.min()),
      float(field_prior.mean()),
      float(field_prior.max()))
