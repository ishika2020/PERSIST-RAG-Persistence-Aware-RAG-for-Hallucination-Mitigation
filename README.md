# PERSIST-RAG: Persistence-Aware Retrieval-Augmented Generation

> A framework for detecting and mitigating **hallucination persistence** in medical-domain LLMs — treating hallucination as a *claim-level stability failure*, not a one-time generation error.

---

## 📌 Overview

Large Language Models (LLMs) augmented with Retrieval-Augmented Generation (RAG) still hallucinate — even when the correct evidence is retrieved. This phenomenon, **hallucination persistence**, occurs when models generate fluent, self-consistent text whose factual claims lack evidentiary support.

**PERSIST-RAG** is a post-generation factual validation framework that:
- Decomposes generated answers into **atomic claims**
- Subjects each claim to **four independent stress tests**
- Assigns a **Claim Persistence Score (CPS)** to govern a three-tier accept/flag/abstain policy
- Requires **no model fine-tuning or retraining** — fully model-agnostic

---

## 🏗️ Architecture

PERSIST-RAG operates in five sequential stages:

```
User Query
    │
    ▼
[1] Uncertainty Estimator (BioMistral-7B NLL)
    ├── Low Uncertainty → Direct Generation
    └── High Uncertainty → Adaptive Hybrid Retrieval
                                │
                                ▼
                        [2] Evidence Pool
                     (Dense: S-PubMedBERT + FAISS)
                     (Sparse: BM25Okapi)
                     (Merged via Reciprocal Rank Fusion)
                                │
                                ▼
                    [3] Grounded Answer Generation
                      (BioMistral-7B, 4-bit NF4)
                                │
                                ▼
                    [4] Atomic Claim Decomposition
                                │
                                ▼
                  [5] Persistence Stress Testing
          ┌─────────────────────┬──────────────────────┐
          │                     │                      │
    Reasoning (R)         Evidence (E)        Counterfactual (C)   Temporal (T)
          │                     │                      │
          └─────────────────────┴──────────────────────┘
                                │
                         CPS Computation
                    CPS = 0.25R + 0.30E + 0.25C + 0.20T
                                │
                ┌───────────────┼───────────────┐
            CPS ≥ 0.65     0.45–0.65        CPS < 0.45
             Accept       Flag + Caveat    Regenerate / Abstain
```

---

## 🔬 Methodology

### 1. Uncertainty-Guided Routing
Estimates epistemic confidence using mean Negative Log-Likelihood (NLL) per token in a single forward pass:

$$u(q) = \text{clip}\left(\frac{-\frac{1}{T} \sum \log P(x_t | x_{<t})}{6.0},\ 0,\ 1\right)$$

Queries with `u(q) ≥ 0.55` route to adaptive retrieval; otherwise direct generation is used.

### 2. Adaptive Hybrid Retrieval
Fuses dense and sparse retrieval over a 5,000-document PubMedQA knowledge base:
- **Dense**: S-PubMedBERT + FAISS cosine nearest-neighbour
- **Sparse**: BM25Okapi (`k₁=1.5`, `b=0.75`)
- **Fusion**: Reciprocal Rank Fusion (RRF, `k=60`) → top-5 documents

### 3. Persistence Stress Testing

| Metric | Description | Model |
|--------|-------------|-------|
| **R — Reasoning** | Mean pairwise cosine similarity across 3 Chain-of-Thought responses | S-PubMedBERT |
| **E — Evidence** | Max NLI entailment probability against top-5 retrieved passages | cross-encoder/nli-MiniLM2-L6-H768 |
| **C — Counterfactual** | NLI contradiction probability against rule-based negations | cross-encoder/nli-MiniLM2-L6-H768 |
| **T — Temporal** | Mean NLI entailment across Wikipedia, pre-2017 PubMed, post-2017 PubMed | cross-encoder/nli-MiniLM2-L6-H768 |

### CPS Formula
```
CPS(cᵢ) = 0.25·R + 0.30·E + 0.25·C + 0.20·T
```
Evidence receives the highest weight (0.30) as NLI entailment from curated medical literature provides the strongest factual ground-truth signal.

---

## 📊 Results

### Factual Quality Comparison

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 | Abstention | Avg CPS |
|--------|---------|---------|---------|--------------|------------|---------|
| **PERSIST-RAG (Ours)** | **0.3405** | **0.1860** | **0.3291** | **0.6547** | 0.80 | **0.2505** |
| Vanilla RAG | 0.1121 | 0.0684 | 0.1139 | 0.5389 | 0.00 | — |
| Direct Generation | 0.1656 | 0.1263 | 0.1661 | 0.5761 | 0.00 | — |

- **+189%** improvement in ROUGE-L over Vanilla RAG
- **+21.5%** improvement in BERTScore-F1 over Vanilla RAG
- The 0.00 abstention rate in baselines reflects unconditional assertion — not accuracy

### CPS Sub-Score Analysis

| Metric | Mean Score |
|--------|-----------|
| Reasoning (R) | 0.943 |
| Temporal (T) | 0.248 |
| Counterfactual (C) | 0.212 |
| Evidence (E) | 0.107 |

> **Key finding**: The R–E divergence (0.943 vs 0.107) is the central empirical result. BioMistral produces highly self-consistent reasoning while its claims lack evidentiary support — this *is* hallucination persistence.

### Ablation Study

| Configuration | Avg CPS | Validation Rate | Abstention Rate |
|---------------|---------|-----------------|-----------------|
| Full PERSIST-RAG | 0.3707 | 0.30 | 0.70 |
| w/o Reasoning (R=0) | 0.1792 | 0.00 | 1.00 |
| w/o Evidence (E=0) | 0.5137 | 1.00 | 0.00 |
| w/o Counterfactual (C=0) | 0.4350 | 0.80 | 0.20 |
| w/o Temporal (T=0) | 0.3894 | 0.40 | 0.60 |

- Removing **E** causes every claim to pass — confirming Evidence Re-Anchoring is the **primary quality gate**
- Removing **R** collapses validation to 0% — confirming Reasoning Diversification is the **most impactful single component**

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Base LLM | BioMistral-7B-DARE (4-bit NF4 quantisation) |
| Dense Retrieval | S-PubMedBERT + FAISS IndexFlatIP |
| Sparse Retrieval | BM25Okapi |
| NLI Model | cross-encoder/nli-MiniLM2-L6-H768 |
| Knowledge Base | PubMedQA (5,000 documents) |
| Benchmark | MedQA-USMLE |
| Evaluation | ROUGE (1/2/L), BERTScore-F1 (DeBERTa-XL-MNLI) |
| Hardware | Google Colab, NVIDIA T4 GPU (16 GB VRAM) |

---

## 📈 Limitations & Future Work

- **Inference latency**: ~60.6 seconds per query; claim-level parallelisation is a priority for future work
- **NLI model domain gap**: General-domain NLI models underperform on medical abstracts (E mean = 0.107); evaluation with **MedNLI / BioNLI** is planned
- **Temporal coverage**: Wikipedia provides limited coverage of specialised clinical topics
- **Scale**: Current evaluation on a MedQA-USMLE subset; expansion to **BioASQ** and full **PubMedQA** test splits is planned


---

## 📜 License

This project is submitted as academic coursework at VIT Vellore. For reuse or collaboration, please contact the authors.
