"""Plots and RESULTS.md / metrics writers for the toy RAG smoke."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def make_plots(
    results: Path,
    hit_at_1: float,
    hit_at_3: float,
    mrr: float,
    rag_acc: float,
    base_acc: float,
    faithfulness: float,
    top_scores: Sequence[float],
) -> List[str]:
    """Write SVG (and PNG) plots; return list of basenames written."""
    written: List[str] = []

    # 1) Hit@k / MRR bar chart
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    labels = ["Hit@1", "Hit@3", "MRR"]
    vals = [hit_at_1, hit_at_3, mrr]
    colors = ["#4c78a8", "#54a24b", "#f58518"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score")
    ax.set_title("Retrieval quality on labeled QA")
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    for ext in ("svg", "png"):
        name = f"hit_at_k.{ext}"
        fig.savefig(results / name, dpi=90 if ext == "png" else None)
        written.append(name)
    plt.close(fig)

    # 2) RAG vs baseline answer accuracy (+ faithfulness)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    labels = ["RAG\n(extractive)", "No-retrieval\nbaseline", "Faithfulness\n(retrieved has gold)"]
    vals = [rag_acc, base_acc, faithfulness]
    colors = ["#54a24b", "#e45756", "#72b7b2"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("accuracy / rate")
    ax.set_title("Answer accuracy: RAG vs no-retrieval")
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    for ext in ("svg", "png"):
        name = f"rag_vs_baseline.{ext}"
        fig.savefig(results / name, dpi=90 if ext == "png" else None)
        written.append(name)
    plt.close(fig)

    # 3) Retrieval score distribution (top-1 cosines over QA)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    scores = np.asarray(list(top_scores), dtype=np.float64)
    ax.hist(scores, bins=min(10, max(5, len(scores))), color="#4c78a8", edgecolor="white", alpha=0.9)
    ax.axvline(float(np.mean(scores)), color="#e45756", ls="--", lw=1.5, label=f"mean={np.mean(scores):.2f}")
    ax.set_xlabel("top-1 cosine similarity")
    ax.set_ylabel("count (QA pairs)")
    ax.set_title("Retrieval score distribution (top-1)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("svg", "png"):
        name = f"score_distribution.{ext}"
        fig.savefig(results / name, dpi=90 if ext == "png" else None)
        written.append(name)
    plt.close(fig)

    return written


def write_artifacts(
    results: Path,
    metrics: Dict[str, Any],
    shot: Dict[str, Any],
    rows: List[Dict[str, Any]],
    plot_names: Sequence[str],
) -> None:
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (results / "JSON.shot").write_text(json.dumps(shot, indent=2) + "\n")

    check_rows = []
    for r in rows:
        status = "PASS" if r["hit_at_3"] and r["rag_contains_gold"] else (
            "HIT" if r["hit_at_3"] else "MISS"
        )
        tops = ", ".join(r["retrieved_ids"][:3])
        check_rows.append(
            f"| `{r['id']}` | {r['question'][:48]} | {tops} | "
            f"{'Y' if r['rag_contains_gold'] else 'N'} / "
            f"{'Y' if r['baseline_contains_gold'] else 'N'} | {status} |"
        )

    svg_plots = [p for p in plot_names if p.endswith(".svg")]
    plot_lines = "\n".join(f"- [`{p}`]({p})" for p in svg_plots)

    md = f"""# Smoke results -- ai-learn-08-rag-scratch

**Seed:** `{metrics['seed']}` | docs={metrics['n_docs']} | vocab={metrics['vocab_size']} | dim={metrics['dim']} | k={metrics['k']}

## Headline metrics

| Metric | Value |
|--------|------:|
| Hit@1 | {metrics['hit_at_1']:.4f} ({metrics['n_hit_at_1']}/{metrics['n_qa']}) |
| Hit@3 | {metrics['hit_at_3']:.4f} ({metrics['n_hit_at_3']}/{metrics['n_qa']}) |
| MRR | {metrics['mrr']:.4f} |
| RAG answer accuracy (contains gold) | {metrics['rag_answer_accuracy']:.4f} ({metrics['n_rag_ok']}/{metrics['n_qa']}) |
| Baseline answer accuracy (no retrieval) | {metrics['baseline_answer_accuracy']:.4f} ({metrics['n_baseline_ok']}/{metrics['n_qa']}) |
| Faithfulness (retrieved contains gold) | {metrics['faithfulness']:.4f} ({metrics['n_faithful']}/{metrics['n_qa']}) |
| RAG beats baseline | {"PASS" if metrics['rag_beats_baseline'] else "FAIL"} |
| Wall time (CPU) | {metrics['runtime_s']:.2f}s |

## Per-query checks

| ID | Question | Top-3 docs | RAG/Base | Status |
|----|----------|------------|----------|--------|
{chr(10).join(check_rows)}

## Plots

{plot_lines}

## Takeaway

Toy RAG = **retrieve** relevant passages (TF-IDF + cosine) then **generate** a
grounded answer by extracting overlapping sentences. Compared with a no-retrieval
baseline that quotes a random document, RAG lands the gold fact far more often
when Hit@k is strong --- the classic reason production assistants retrieve before
they answer.
"""
    (results / "RESULTS.md").write_text(md)
