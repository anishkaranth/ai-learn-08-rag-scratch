# AI Learn 08 — Toy RAG from Scratch

Build a **pure-NumPy RAG loop** over a toy FAQ corpus: TF-IDF retrieval, extractive / template “generation”, and eval that shows grounded retrieval beats a no-retrieval baseline.

Phase B (AI components) — follows similarity search (`ai-learn-07`). Self-contained; does not import that repo.

## Learning goals

- **RAG loop**: retrieve top-k passages → generate an answer grounded in those passages
- **TF-IDF + L2 cosine** as a classical sparse retriever (same idea as `ai-learn-07`)
- **Extractive generation** without an LLM API: stitch query-overlapping sentences from retrieved docs
- **Eval**: Hit@1 / Hit@3 / MRR for retrieval; answer contains-gold + faithfulness; compare vs no-retrieval baseline
- Why grounding matters: random / fixed answers almost never hit the gold fact

## Brief math

TF-IDF for term \(t\) in document \(d\):

\[
\mathrm{tfidf}(t,d) = (1 + \log \mathrm{tf}_{t,d}) \cdot \left(\log\frac{N+1}{\mathrm{df}_t+1} + 1\right)
\]

Cosine on L2-normalized vectors \(\hat{q}, \hat{d}\): \(\cos = \hat{q}^\top \hat{d}\).

Toy RAG:

\[
\mathcal{C} = \mathrm{TopK}(q; \mathcal{D}), \quad
\hat{a} = \mathrm{Extract}(q, \mathcal{C})
\]

where \(\mathrm{Extract}\) picks sentences in \(\mathcal{C}\) with highest query-term overlap.

## Layout

```
corpus.py              # FAQ corpus + labeled QA pairs
rag.py                 # TF-IDF, retrieve, generate, metrics
smoke_plots.py         # SVG plots + RESULTS.md writers
run_smoke.py           # end-to-end smoke -> results/
notebooks/rag_scratch.ipynb
results/               # committed metrics + plots
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```

Runs on CPU in a few seconds. See `results/RESULTS.md` for the latest smoke metrics.

## What you'll learn next (Phase B)

Tool-calling stubs, eval harnesses, and light fine-tuning (e.g. LoRA) — then a full end-to-end AI app milestone.
