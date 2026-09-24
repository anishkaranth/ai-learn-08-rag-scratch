"""Toy multi-topic FAQ corpus and labeled QA pairs for toy RAG."""
from __future__ import annotations

from typing import Dict, List

# Short FAQ / doc snippets across a few topics. Keep facts crisp so extractive
# generation can pull a gold span, and TF-IDF retrieval can hit the right doc.
DOCUMENTS: List[Dict[str, str]] = [
    {
        "id": "py1",
        "topic": "python",
        "text": (
            "Python is a high-level programming language created by Guido van Rossum. "
            "It emphasizes readability and uses significant indentation."
        ),
    },
    {
        "id": "py2",
        "topic": "python",
        "text": (
            "A Python list is an ordered mutable sequence. "
            "You can append items with list.append and slice with bracket notation."
        ),
    },
    {
        "id": "py3",
        "topic": "python",
        "text": (
            "Python dictionaries map keys to values. "
            "Lookup by key is average O(1) time because dicts use a hash table."
        ),
    },
    {
        "id": "py4",
        "topic": "python",
        "text": (
            "Virtual environments isolate Python package installs per project. "
            "Create one with python -m venv .venv then activate it before pip install."
        ),
    },
    {
        "id": "ml1",
        "topic": "ml",
        "text": (
            "Supervised learning trains on labeled examples. "
            "Common tasks are classification for discrete labels and regression for continuous targets."
        ),
    },
    {
        "id": "ml2",
        "topic": "ml",
        "text": (
            "Overfitting means a model fits training noise and fails to generalize. "
            "Regularization, early stopping, and more data help reduce overfitting."
        ),
    },
    {
        "id": "ml3",
        "topic": "ml",
        "text": (
            "Gradient descent updates parameters by subtracting a learning rate times the loss gradient. "
            "Mini-batch SGD approximates the full gradient on random subsets."
        ),
    },
    {
        "id": "ml4",
        "topic": "ml",
        "text": (
            "Cross-validation splits data into folds to estimate generalization. "
            "K-fold cross-validation trains on K-1 folds and validates on the held-out fold."
        ),
    },
    {
        "id": "emb1",
        "topic": "embeddings",
        "text": (
            "Word embeddings map tokens to dense vectors that capture semantic similarity. "
            "Cosine similarity between embedding vectors measures relatedness."
        ),
    },
    {
        "id": "emb2",
        "topic": "embeddings",
        "text": (
            "TF-IDF weights rare informative terms higher than frequent stopwords. "
            "A TF-IDF document vector is useful for classical retrieval baselines."
        ),
    },
    {
        "id": "emb3",
        "topic": "embeddings",
        "text": (
            "Skip-gram with negative sampling learns embeddings by predicting context words. "
            "Negative samples push unrelated word pairs apart in vector space."
        ),
    },
    {
        "id": "rag1",
        "topic": "rag",
        "text": (
            "Retrieval-Augmented Generation (RAG) first retrieves relevant documents then generates an answer. "
            "Grounding generation on retrieved context reduces hallucination."
        ),
    },
    {
        "id": "rag2",
        "topic": "rag",
        "text": (
            "A RAG pipeline has a retriever and a generator. "
            "The retriever returns top-k passages; the generator conditions on those passages."
        ),
    },
    {
        "id": "rag3",
        "topic": "rag",
        "text": (
            "Faithfulness means the answer is supported by retrieved documents. "
            "Eval often checks whether gold facts appear in the retrieved context."
        ),
    },
    {
        "id": "rag4",
        "topic": "rag",
        "text": (
            "Hybrid retrieval can combine sparse TF-IDF with dense embedding search. "
            "Re-ranking top candidates with a cross-encoder often improves precision."
        ),
    },
    {
        "id": "nlp1",
        "topic": "nlp",
        "text": (
            "Tokenization splits text into words or subword units for modeling. "
            "Byte-pair encoding (BPE) builds a subword vocabulary from frequent merges."
        ),
    },
    {
        "id": "nlp2",
        "topic": "nlp",
        "text": (
            "Transformers use self-attention to mix information across token positions. "
            "Multi-head attention runs several attention patterns in parallel."
        ),
    },
    {
        "id": "nlp3",
        "topic": "nlp",
        "text": (
            "Positional encodings inject order information into transformer inputs. "
            "Sinusoidal encodings use sine and cosine of different frequencies."
        ),
    },
    {
        "id": "wx1",
        "topic": "weather",
        "text": (
            "Photosynthesis converts sunlight carbon dioxide and water into glucose and oxygen. "
            "This process occurs mainly in plant chloroplasts."
        ),
    },
    {
        "id": "wx2",
        "topic": "weather",
        "text": (
            "A cold front brings cooler denser air that often triggers thunderstorms. "
            "Warm fronts typically bring steady light precipitation over a longer period."
        ),
    },
    {
        "id": "wx3",
        "topic": "weather",
        "text": (
            "The water cycle includes evaporation condensation and precipitation. "
            "Clouds form when moist air cools and water vapor condenses into droplets."
        ),
    },
    {
        "id": "bio1",
        "topic": "biology",
        "text": (
            "DNA stores genetic information as sequences of four bases A T C and G. "
            "Genes are transcribed into RNA then translated into proteins."
        ),
    },
    {
        "id": "bio2",
        "topic": "biology",
        "text": (
            "Mitochondria are organelles that produce ATP through cellular respiration. "
            "They are often called the powerhouse of the cell."
        ),
    },
    {
        "id": "cs1",
        "topic": "systems",
        "text": (
            "A hash table maps keys to values using a hash function into buckets. "
            "Average lookup insert and delete are O(1) with a good hash and load factor."
        ),
    },
    {
        "id": "cs2",
        "topic": "systems",
        "text": (
            "HTTP is a request-response protocol used by the web. "
            "Common methods include GET for reads and POST for creating resources."
        ),
    },
    {
        "id": "cs3",
        "topic": "systems",
        "text": (
            "Git tracks file history with commits in a directed acyclic graph. "
            "Branches point to commits and merging combines histories."
        ),
    },
]

# question -> gold doc id(s) + short gold answer fact/span present in those docs
QA_PAIRS: List[Dict[str, object]] = [
    {
        "id": "q01",
        "question": "Who created the Python programming language?",
        "gold_doc_ids": ["py1"],
        "gold_answer": "Guido van Rossum",
    },
    {
        "id": "q02",
        "question": "How do you append an item to a Python list?",
        "gold_doc_ids": ["py2"],
        "gold_answer": "list.append",
    },
    {
        "id": "q03",
        "question": "Why is Python dictionary key lookup fast?",
        "gold_doc_ids": ["py3"],
        "gold_answer": "hash table",
    },
    {
        "id": "q04",
        "question": "How do you create a Python virtual environment?",
        "gold_doc_ids": ["py4"],
        "gold_answer": "python -m venv",
    },
    {
        "id": "q05",
        "question": "What are common supervised learning tasks?",
        "gold_doc_ids": ["ml1"],
        "gold_answer": "classification",
    },
    {
        "id": "q06",
        "question": "What is overfitting in machine learning?",
        "gold_doc_ids": ["ml2"],
        "gold_answer": "fails to generalize",
    },
    {
        "id": "q07",
        "question": "How does gradient descent update parameters?",
        "gold_doc_ids": ["ml3"],
        "gold_answer": "learning rate",
    },
    {
        "id": "q08",
        "question": "What is k-fold cross-validation?",
        "gold_doc_ids": ["ml4"],
        "gold_answer": "held-out fold",
    },
    {
        "id": "q09",
        "question": "How do word embeddings measure relatedness?",
        "gold_doc_ids": ["emb1"],
        "gold_answer": "cosine similarity",
    },
    {
        "id": "q10",
        "question": "What does TF-IDF do to rare terms?",
        "gold_doc_ids": ["emb2"],
        "gold_answer": "weights rare informative terms higher",
    },
    {
        "id": "q11",
        "question": "What is Retrieval-Augmented Generation?",
        "gold_doc_ids": ["rag1"],
        "gold_answer": "retrieves relevant documents then generates",
    },
    {
        "id": "q12",
        "question": "What are the two main parts of a RAG pipeline?",
        "gold_doc_ids": ["rag2"],
        "gold_answer": "retriever and a generator",
    },
    {
        "id": "q13",
        "question": "What does faithfulness mean in RAG evaluation?",
        "gold_doc_ids": ["rag3"],
        "gold_answer": "supported by retrieved documents",
    },
    {
        "id": "q14",
        "question": "What tokenization method builds a subword vocabulary?",
        "gold_doc_ids": ["nlp1"],
        "gold_answer": "Byte-pair encoding",
    },
    {
        "id": "q15",
        "question": "What mechanism do transformers use across token positions?",
        "gold_doc_ids": ["nlp2"],
        "gold_answer": "self-attention",
    },
    {
        "id": "q16",
        "question": "What do sinusoidal positional encodings use?",
        "gold_doc_ids": ["nlp3"],
        "gold_answer": "sine and cosine",
    },
    {
        "id": "q17",
        "question": "Where does photosynthesis mainly occur in plants?",
        "gold_doc_ids": ["wx1"],
        "gold_answer": "chloroplasts",
    },
    {
        "id": "q18",
        "question": "What weather often follows a cold front?",
        "gold_doc_ids": ["wx2"],
        "gold_answer": "thunderstorms",
    },
    {
        "id": "q19",
        "question": "What are mitochondria often called?",
        "gold_doc_ids": ["bio2"],
        "gold_answer": "powerhouse of the cell",
    },
    {
        "id": "q20",
        "question": "What average complexity does a hash table lookup have?",
        "gold_doc_ids": ["cs1"],
        "gold_answer": "O(1)",
    },
    {
        "id": "q21",
        "question": "Which HTTP method is commonly used for reads?",
        "gold_doc_ids": ["cs2"],
        "gold_answer": "GET",
    },
    {
        "id": "q22",
        "question": "How does Git store commit history structure?",
        "gold_doc_ids": ["cs3"],
        "gold_answer": "directed acyclic graph",
    },
]
