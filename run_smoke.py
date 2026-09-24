#!/usr/bin/env python3
"""Toy RAG smoke: TF-IDF retrieve + extractive generate -> results/."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from corpus import DOCUMENTS, QA_PAIRS
from rag import (
    build_vocab,
    compute_idf,
    evaluate_qa,
    l2_normalize,
    tfidf_matrix,
)
from smoke_plots import make_plots, write_artifacts

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
K = 3


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    word2id, id2word = build_vocab(DOCUMENTS)
    idf = compute_idf(DOCUMENTS, word2id)
    X = l2_normalize(tfidf_matrix(DOCUMENTS, word2id, idf))
    dim = int(X.shape[1])
    n_docs = int(X.shape[0])

    eval_out = evaluate_qa(word2id, idf, X, QA_PAIRS, DOCUMENTS, k=K, seed=SEED)
    rows = eval_out["rows"]  # type: ignore[assignment]
    top_scores = [float(r["top_cosines"][0]) for r in rows if r["top_cosines"]]

    plot_names = make_plots(
        RESULTS,
        hit_at_1=float(eval_out["hit_at_1"]),
        hit_at_3=float(eval_out["hit_at_3"]),
        mrr=float(eval_out["mrr"]),
        rag_acc=float(eval_out["rag_answer_accuracy"]),
        base_acc=float(eval_out["baseline_answer_accuracy"]),
        faithfulness=float(eval_out["faithfulness"]),
        top_scores=top_scores,
    )
    runtime_s = time.perf_counter() - t0

    rag_beats = float(eval_out["rag_answer_accuracy"]) > float(
        eval_out["baseline_answer_accuracy"]
    ) + 0.2

    metrics = {
        "project": "ai-learn-08-rag-scratch",
        "seed": SEED,
        "n_docs": n_docs,
        "vocab_size": len(id2word),
        "dim": dim,
        "k": K,
        "n_qa": int(eval_out["n_qa"]),
        "hit_at_1": round(float(eval_out["hit_at_1"]), 4),
        "hit_at_3": round(float(eval_out["hit_at_3"]), 4),
        "mrr": round(float(eval_out["mrr"]), 4),
        "rag_answer_accuracy": round(float(eval_out["rag_answer_accuracy"]), 4),
        "baseline_answer_accuracy": round(float(eval_out["baseline_answer_accuracy"]), 4),
        "faithfulness": round(float(eval_out["faithfulness"]), 4),
        "n_hit_at_1": int(eval_out["n_hit_at_1"]),
        "n_hit_at_3": int(eval_out["n_hit_at_3"]),
        "n_rag_ok": int(eval_out["n_rag_ok"]),
        "n_baseline_ok": int(eval_out["n_baseline_ok"]),
        "n_faithful": int(eval_out["n_faithful"]),
        "rag_beats_baseline": bool(rag_beats),
        "runtime_s": round(runtime_s, 3),
        "per_query": [
            {
                "id": r["id"],
                "question": r["question"],
                "gold_doc_ids": r["gold_doc_ids"],
                "retrieved_ids": r["retrieved_ids"],
                "top_cosines": r["top_cosines"],
                "hit_at_1": r["hit_at_1"],
                "hit_at_3": r["hit_at_3"],
                "rr": r["rr"],
                "faithfulness": r["faithfulness"],
                "rag_contains_gold": r["rag_contains_gold"],
                "baseline_contains_gold": r["baseline_contains_gold"],
                "rag_answer": r["rag_answer"],
                "baseline_answer": r["baseline_answer"],
            }
            for r in rows
        ],
    }
    shot = {
        "project": metrics["project"],
        "n_docs": n_docs,
        "vocab_size": len(id2word),
        "dim": dim,
        "k": K,
        "n_qa": metrics["n_qa"],
        "hit_at_1": metrics["hit_at_1"],
        "hit_at_3": metrics["hit_at_3"],
        "mrr": metrics["mrr"],
        "rag_answer_accuracy": metrics["rag_answer_accuracy"],
        "baseline_answer_accuracy": metrics["baseline_answer_accuracy"],
        "faithfulness": metrics["faithfulness"],
        "rag_beats_baseline": metrics["rag_beats_baseline"],
        "runtime_s": metrics["runtime_s"],
        "seed": SEED,
    }
    write_artifacts(RESULTS, metrics, shot, rows, plot_names)
    print(json.dumps(shot, indent=2))
    print(
        f"\nWrote results/ in {runtime_s:.2f}s -- "
        f"Hit@1={metrics['hit_at_1']:.3f} Hit@3={metrics['hit_at_3']:.3f} "
        f"RAG={metrics['rag_answer_accuracy']:.3f} "
        f"baseline={metrics['baseline_answer_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
