import os, time, math, datetime as dt, faiss, numpy as np
import tiktoken, openai, requests
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------- Embeddings ----------
EMBED_MODEL = "text-embedding-3-small"

def embed(texts: list[str]) -> np.ndarray:
    """Return NxD numpy array of embeddings for a list of strings."""
    resp = openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data]).astype("float32")

# ---------- Simple chunking ----------
def chunk(text: str, max_tokens: int = 800) -> list[str]:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")  # any encoding works
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = start + max_tokens
        chunks.append(enc.decode(tokens[start:end]))
        start = end
    return chunks

# ---------- Time-decay scoring ----------
def time_decay(pub_date: str, half_life_weeks: int = 12) -> float:
    """Return multiplier ∈ (0,1] given ISO publish date."""
    weeks_old = (dt.date.today() - dt.date.fromisoformat(pub_date[:10])).days / 7
    return 0.5 ** (weeks_old / half_life_weeks)

# ---------- FAISS helpers ----------
def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vectors.shape[1])
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    index.add(vectors / norm)                          # cosine sim = dot on unit sphere
    return index

def search(index, query_emb: np.ndarray, k: int = 6):
    faiss.normalize_L2(query_emb)
    sims, idx = index.search(query_emb, k)
    return idx[0], sims[0]
