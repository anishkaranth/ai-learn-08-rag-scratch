# Smoke results -- ai-learn-08-rag-scratch

**Seed:** `42` | docs=26 | vocab=353 | dim=353 | k=3

## Headline metrics

| Metric | Value |
|--------|------:|
| Hit@1 | 1.0000 (22/22) |
| Hit@3 | 1.0000 (22/22) |
| MRR | 1.0000 |
| RAG answer accuracy (contains gold) | 1.0000 (22/22) |
| Baseline answer accuracy (no retrieval) | 0.0000 (0/22) |
| Faithfulness (retrieved contains gold) | 1.0000 (22/22) |
| RAG beats baseline | PASS |
| Wall time (CPU) | 0.02s |

## Per-query checks

| ID | Question | Top-3 docs | RAG/Base | Status |
|----|----------|------------|----------|--------|
| `q01` | Who created the Python programming language? | py1, py4, bio2 | Y / N | PASS |
| `q02` | How do you append an item to a Python list? | py2, py3, py4 | Y / N | PASS |
| `q03` | Why is Python dictionary key lookup fast? | py3, py1, py2 | Y / N | PASS |
| `q04` | How do you create a Python virtual environment? | py4, py2, py3 | Y / N | PASS |
| `q05` | What are common supervised learning tasks? | ml1, bio2, cs2 | Y / N | PASS |
| `q06` | What is overfitting in machine learning? | ml2, rag3, ml1 | Y / N | PASS |
| `q07` | How does gradient descent update parameters? | ml3, py1, py3 | Y / N | PASS |
| `q08` | What is k-fold cross-validation? | ml4, rag4, rag2 | Y / N | PASS |
| `q09` | How do word embeddings measure relatedness? | emb1, emb3, py1 | Y / N | PASS |
| `q10` | What does TF-IDF do to rare terms? | emb2, rag4, py3 | Y / N | PASS |
| `q11` | What is Retrieval-Augmented Generation? | rag1, emb2, rag4 | Y / N | PASS |
| `q12` | What are the two main parts of a RAG pipeline? | rag2, bio2, bio1 | Y / N | PASS |
| `q13` | What does faithfulness mean in RAG evaluation? | rag3, rag1, rag2 | Y / N | PASS |
| `q14` | What tokenization method builds a subword vocabu | nlp1, rag2, cs1 | Y / N | PASS |
| `q15` | What mechanism do transformers use across token  | nlp2, py3, nlp3 | Y / N | PASS |
| `q16` | What do sinusoidal positional encodings use? | nlp3, py3, nlp2 | Y / N | PASS |
| `q17` | Where does photosynthesis mainly occur in plants | wx1, rag3, cs3 | Y / N | PASS |
| `q18` | What weather often follows a cold front? | wx2, rag4, bio2 | Y / N | PASS |
| `q19` | What are mitochondria often called? | bio2, ml1, rag3 | Y / N | PASS |
| `q20` | What average complexity does a hash table lookup | cs1, py3, rag2 | Y / N | PASS |
| `q21` | Which HTTP method is commonly used for reads? | cs2, ml1, emb2 | Y / N | PASS |
| `q22` | How does Git store commit history structure? | cs3, py1, py3 | Y / N | PASS |

## Plots

- [`hit_at_k.svg`](hit_at_k.svg)
- [`rag_vs_baseline.svg`](rag_vs_baseline.svg)
- [`score_distribution.svg`](score_distribution.svg)

## Takeaway

Toy RAG = **retrieve** relevant passages (TF-IDF + cosine) then **generate** a
grounded answer by extracting overlapping sentences. Compared with a no-retrieval
baseline that quotes a random document, RAG lands the gold fact far more often
when Hit@k is strong --- the classic reason production assistants retrieve before
they answer.
