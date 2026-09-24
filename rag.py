"""Toy RAG from scratch: TF-IDF retrieve + extractive / template generate.

Educational NumPy implementation (no external LLM APIs):
  - TF-IDF bag-of-words vectors, L2-normalized for cosine retrieval
  - Exact top-k retrieval by cosine similarity
  - Extractive generator: pick query-overlapping sentences from retrieved docs
  - No-retrieval baseline: fixed phrase or random-doc template answer
  - Metrics: Hit@k, MRR, faithfulness, answer contains/EM vs gold
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

from corpus import DOCUMENTS, QA_PAIRS

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def build_vocab(docs: Sequence[Dict[str, str]]) -> Tuple[Dict[str, int], List[str]]:
    counts: Dict[str, int] = {}
    for d in docs:
        for tok in set(tokenize(d["text"])):
            counts[tok] = counts.get(tok, 0) + 1
    words = sorted(counts.keys())
    word2id = {w: i for i, w in enumerate(words)}
    return word2id, words


def compute_idf(docs: Sequence[Dict[str, str]], word2id: Dict[str, int]) -> np.ndarray:
    """Smooth IDF: log((N + 1) / (df + 1)) + 1."""
    n = len(docs)
    df = np.zeros(len(word2id), dtype=np.float64)
    for d in docs:
        seen = set(tokenize(d["text"]))
        for tok in seen:
            if tok in word2id:
                df[word2id[tok]] += 1.0
    return np.log((n + 1.0) / (df + 1.0)) + 1.0


def tfidf_matrix(
    docs: Sequence[Dict[str, str]],
    word2id: Dict[str, int],
    idf: np.ndarray,
) -> np.ndarray:
    """Return (N, V) log-TF * IDF matrix (not yet L2-normalized)."""
    n, v = len(docs), len(word2id)
    X = np.zeros((n, v), dtype=np.float64)
    for i, d in enumerate(docs):
        toks = tokenize(d["text"])
        for tok in toks:
            if tok in word2id:
                X[i, word2id[tok]] += 1.0
        mask = X[i] > 0
        X[i, mask] = 1.0 + np.log(X[i, mask])
        X[i] *= idf
    return X


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def embed_query(text: str, word2id: Dict[str, int], idf: np.ndarray) -> np.ndarray:
    """TF-IDF vector for a free-text query, L2-normalized."""
    v = np.zeros(len(word2id), dtype=np.float64)
    toks = tokenize(text)
    for tok in toks:
        if tok in word2id:
            v[word2id[tok]] += 1.0
    mask = v > 0
    v[mask] = 1.0 + np.log(v[mask])
    v *= idf
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def exact_topk(
    query_vec: np.ndarray,
    doc_matrix: np.ndarray,
    k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of top-k docs by cosine, descending."""
    scores = doc_matrix @ query_vec
    k = min(k, len(scores))
    part = np.argpartition(-scores, kth=k - 1)[:k]
    order = np.argsort(-scores[part])
    idx = part[order]
    return idx, scores[idx]


def doc_id_to_index(docs: Sequence[Dict[str, str]] = DOCUMENTS) -> Dict[str, int]:
    return {d["id"]: i for i, d in enumerate(docs)}


def retrieve(
    question: str,
    word2id: Dict[str, int],
    idf: np.ndarray,
    doc_matrix: np.ndarray,
    docs: Sequence[Dict[str, str]] = DOCUMENTS,
    k: int = 3,
) -> List[Dict[str, object]]:
    """TF-IDF exact cosine top-k retrieval."""
    qv = embed_query(question, word2id, idf)
    idx, scores = exact_topk(qv, doc_matrix, k=k)
    hits = []
    for rank, (i, s) in enumerate(zip(idx, scores), start=1):
        d = docs[int(i)]
        hits.append(
            {
                "rank": rank,
                "index": int(i),
                "doc_id": d["id"],
                "topic": d["topic"],
                "cosine": float(s),
                "text": d["text"],
            }
        )
    return hits


# ---------------------------------------------------------------------------
# Toy generators
# ---------------------------------------------------------------------------

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "and", "or", "with", "by", "from",
    "what", "who", "how", "why", "when", "where", "which", "do", "does",
    "did", "you", "your", "it", "its", "this", "that", "these", "those",
}


def _query_content_tokens(question: str) -> set:
    return {t for t in tokenize(question) if t not in _STOP and len(t) > 1}


def _sentence_overlap_score(sentence: str, q_toks: set) -> float:
    s_toks = set(tokenize(sentence))
    if not q_toks or not s_toks:
        return 0.0
    return len(q_toks & s_toks) / len(q_toks)


def extractive_generate(
    question: str,
    retrieved: Sequence[Dict[str, object]],
    max_sentences: int = 2,
) -> str:
    """Stitch top overlapping sentences from retrieved passages into an answer."""
    q_toks = _query_content_tokens(question)
    scored: List[Tuple[float, str, str]] = []
    for hit in retrieved:
        text = str(hit["text"])
        sents = _SENT_SPLIT_RE.split(text.strip())
        for sent in sents:
            sent = sent.strip()
            if not sent:
                continue
            score = _sentence_overlap_score(sent, q_toks)
            # slight boost by retrieval rank (rank 1 preferred)
            rank_boost = 0.05 * max(0, 4 - int(hit["rank"]))
            scored.append((score + rank_boost, sent, str(hit["doc_id"])))
    if not scored:
        return "I could not find a grounded answer in the retrieved documents."
    scored.sort(key=lambda x: -x[0])
    # de-duplicate near-identical sentences
    chosen: List[str] = []
    used_docs: List[str] = []
    for score, sent, doc_id in scored:
        if score <= 0 and chosen:
            break
        norm = sent.lower()
        if any(norm == c.lower() for c in chosen):
            continue
        chosen.append(sent if sent.endswith((".", "!", "?")) else sent + ".")
        used_docs.append(doc_id)
        if len(chosen) >= max_sentences:
            break
    if not chosen:
        # fall back to first retrieved sentence
        first = str(retrieved[0]["text"]).strip()
        sents = _SENT_SPLIT_RE.split(first)
        chosen = [sents[0].strip()]
        used_docs = [str(retrieved[0]["doc_id"])]
        if chosen[0] and not chosen[0].endswith((".", "!", "?")):
            chosen[0] += "."
    sources = ", ".join(dict.fromkeys(used_docs))
    body = " ".join(chosen)
    return f"Based on [{sources}]: {body}"


def baseline_generate(
    question: str,
    docs: Sequence[Dict[str, str]] = DOCUMENTS,
    rng: np.random.Generator | None = None,
    mode: str = "random_doc",
) -> str:
    """No-retrieval baseline: fixed phrase or random document snippet."""
    if mode == "fixed":
        return (
            "I do not have retrieved context, so I cannot ground an answer. "
            "Please consult the knowledge base."
        )
    if rng is None:
        rng = np.random.default_rng(0)
    d = docs[int(rng.integers(0, len(docs)))]
    sents = _SENT_SPLIT_RE.split(d["text"].strip())
    snippet = sents[0].strip()
    if snippet and not snippet.endswith((".", "!", "?")):
        snippet += "."
    return f"Without retrieval (random doc {d['id']}): {snippet}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def answer_contains(pred: str, gold: str) -> bool:
    """True if normalized gold span appears in normalized prediction."""
    g = normalize_answer(gold)
    p = normalize_answer(pred)
    if not g:
        return False
    return g in p


def answer_em(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def faithfulness(retrieved: Sequence[Dict[str, object]], gold_answer: str) -> bool:
    """Gold fact appears somewhere in the retrieved document texts."""
    blob = " ".join(str(h["text"]) for h in retrieved)
    return answer_contains(blob, gold_answer)


def hit_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> bool:
    top = set(list(retrieved_ids)[:k])
    return bool(top & set(gold_ids))


def reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def evaluate_qa(
    word2id: Dict[str, int],
    idf: np.ndarray,
    doc_matrix: np.ndarray,
    qa_pairs: Sequence[Dict[str, object]] = QA_PAIRS,
    docs: Sequence[Dict[str, str]] = DOCUMENTS,
    k: int = 3,
    seed: int = 42,
) -> Dict[str, object]:
    """Run retrieval + RAG generate + baseline; aggregate headline metrics."""
    rng = np.random.default_rng(seed)
    rows = []
    hits1 = hits3 = 0
    mrr_sum = 0.0
    rag_ok = base_ok = faith_ok = 0
    n = len(qa_pairs)

    for qa in qa_pairs:
        q = str(qa["question"])
        gold_ids = list(qa["gold_doc_ids"])  # type: ignore[arg-type]
        gold_ans = str(qa["gold_answer"])
        hits = retrieve(q, word2id, idf, doc_matrix, docs, k=k)
        retrieved_ids = [str(h["doc_id"]) for h in hits]
        h1 = hit_at_k(retrieved_ids, gold_ids, 1)
        h3 = hit_at_k(retrieved_ids, gold_ids, min(3, k))
        rr = reciprocal_rank(retrieved_ids, gold_ids)
        hits1 += int(h1)
        hits3 += int(h3)
        mrr_sum += rr

        rag_ans = extractive_generate(q, hits)
        base_ans = baseline_generate(q, docs, rng=rng, mode="random_doc")
        rag_match = answer_contains(rag_ans, gold_ans)
        base_match = answer_contains(base_ans, gold_ans)
        faith = faithfulness(hits, gold_ans)
        rag_ok += int(rag_match)
        base_ok += int(base_match)
        faith_ok += int(faith)

        rows.append(
            {
                "id": qa["id"],
                "question": q,
                "gold_doc_ids": gold_ids,
                "gold_answer": gold_ans,
                "retrieved_ids": retrieved_ids,
                "top_cosines": [round(float(h["cosine"]), 4) for h in hits],
                "hit_at_1": h1,
                "hit_at_3": h3,
                "rr": round(rr, 4),
                "faithfulness": faith,
                "rag_answer": rag_ans,
                "baseline_answer": base_ans,
                "rag_contains_gold": rag_match,
                "baseline_contains_gold": base_match,
            }
        )

    return {
        "n_qa": n,
        "k": k,
        "hit_at_1": hits1 / n,
        "hit_at_3": hits3 / n,
        "mrr": mrr_sum / n,
        "rag_answer_accuracy": rag_ok / n,
        "baseline_answer_accuracy": base_ok / n,
        "faithfulness": faith_ok / n,
        "n_hit_at_1": hits1,
        "n_hit_at_3": hits3,
        "n_rag_ok": rag_ok,
        "n_baseline_ok": base_ok,
        "n_faithful": faith_ok,
        "rows": rows,
    }
