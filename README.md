# Retrieval-Augmented Question Answering (RAG) · open-source stack

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmedbouzidi304/rag-question-answering/blob/main/rag_question_answering.ipynb)
![Python](https://img.shields.io/badge/Python-3.11-blue) ![FAISS](https://img.shields.io/badge/FAISS-vector%20search-orange) ![Transformers](https://img.shields.io/badge/HF-Transformers-yellow) ![LLM](https://img.shields.io/badge/LLM-Qwen2.5--1.5B-green)

A complete RAG system built and **measured** stage by stage: four retrievers are benchmarked, then an open LLM answers questions with and without retrieval, and every error is attributed to either the retriever or the generator. No API keys: everything runs on a free Colab T4 GPU.

## Architecture
```
question ──► BM25 (sparse) ──┐
         └─► bge-small + FAISS (dense) ─► Reciprocal Rank Fusion ─► cross-encoder re-rank ─► top-3 passages
                                                                                              │
                                                       Qwen2.5-1.5B-Instruct (grounded prompt) ◄┘ ─► answer + sources
```

## Pipeline
| Step | What is done |
|---|---|
| 1. Knowledge base | 2,000+ Wikipedia paragraphs from SQuAD v1.1 (validation), with gold question → passage labels |
| 2. Evaluation design | seeded sample: 1,000 questions for retrieval, 300 for generation |
| 3. Sparse retrieval | BM25 with custom tokenizer |
| 4. Dense retrieval | `BAAI/bge-small-en-v1.5` embeddings in a FAISS inner-product index |
| 5. Hybrid | Reciprocal Rank Fusion of BM25 + dense |
| 6. Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top-20 candidates |
| 7. Retrieval benchmark | Recall@1/3/5/10, MRR@10, latency |
| 8. Generation | `Qwen/Qwen2.5-1.5B-Instruct`, extractive prompt with abstention, batched greedy decoding |
| 9. End-to-end eval | closed-book vs RAG vs oracle, official SQuAD Exact Match / F1 |
| 10. Error analysis | retrieval failure vs generation failure, with examples |
| 11. Demo | `ask()` returns the answer with cited sources, and abstains on out-of-scope questions |

## Results
<!-- RESULTS -->
_Run the notebook to reproduce: results are printed in the benchmark sections._

## Run it
Click **Open in Colab**, select *Runtime → Change runtime type → T4 GPU*, then *Run all* (about 10-15 minutes).

## Tech stack
Hugging Face Transformers & Datasets · sentence-transformers · FAISS · rank-bm25 · PyTorch · pandas · matplotlib
