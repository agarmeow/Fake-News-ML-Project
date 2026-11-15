# Multimodal & Multilingual Misinformation Detection and Verification System

An end-to-end multimodal and cross-lingual fact-checking and misinformation detection framework supporting English, Hindi, and Marathi text, speech, and images. The system integrates OCR and Whisper speech transcription, multilingual sentence embeddings (MuRIL), multimodal vision-language alignment (CLIP), fine-tuned RoBERTa fake-type classification (ordinary vs. satire vs. clickbait vs. misleading), and a multi-stage evidence retrieval and verification engine combining Natural Language Inference (NLI), relational consistency checking, competing-fact search, and Groq-powered LLM arbitration.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Input: Text / Image / Audio     |
                      +---------------------------------------+
                                          |
                      +---------------------------------------+
                      |      Preprocessing & Extraction       |
                      |  - OCR (EasyOCR / Tesseract)          |
                      |  - Speech Transcription (Whisper)     |
                      |  - Claim Normalization & NER (spaCy)  |
                      +---------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
     +-------------------------------+         +-------------------------------+
     |   Multimodal Classification   |         |    Hybrid Fake-Type Engine    |
     |  - Text-Image Alignment (CLIP)|         |  - RoBERTa Fine-Tuned Model   |
     |  - Cross-Lingual (MuRIL)      |         |  - LLM Fallback (Groq LLaMA)  |
     +-------------------------------+         +-------------------------------+
                     \                                         /
                      \                                       /
                       +-------------------------------------+
                       |     Web Evidence Retrieval          |
                       |  - DuckDuckGo Search API + Scraping |
                       |  - Domain Trust & Semantic Rerank   |
                       |  - Robust SHA1 Disk Cache (cache/)  |
                       +-------------------------------------+
                                          |
                       +-------------------------------------+
                       |   Multi-Stage Verification Engine   |
                       |  - DeBERTa NLI Stance Classification|
                       |  - Relational Consistency Checker   |
                       |  - Competing-Fact Bidirectional Pass|
                       |  - LLM Arbitration with Retry / S1 |
                       +-------------------------------------+
                                          |
                       +-------------------------------------+
                       |    Verdict & Confidence Synthesis   |
                       |    TRUE / FALSE / UNVERIFIABLE      |
                       |    + Source Attribution & Conflicts |
                       +-------------------------------------+
```

---

## Requirements

- **Python**: 3.10+ (tested and verified on Python 3.11 and 3.14 on Windows/Linux)
- **Hardware**: GPU with CUDA acceleration recommended for model inference (e.g., CUDA 12.8); runs fully on CPU with fallback.
- **Tesseract OCR** (optional, if using Tesseract backend for image OCR).

---

## Setup & Installation

1. **Clone repository and set up virtual environment**:
   ```bash
   git clone <repo-url>
   cd Fake-News-ML-Project
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > **PyTorch / CUDA Note**: The `requirements.txt` pins `torch==2.11.0+cu128`. If installing on CPU or a different CUDA version, install PyTorch following official instructions at [pytorch.org](https://pytorch.org/get-started/locally/), e.g.:
   > ```bash
   > pip install torch --index-url https://download.pytorch.org/whl/cpu
   > ```

3. **Download spaCy language model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Environment Configuration**:
   Copy the example environment file and add your Groq API credentials:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to supply `GROQ_API_KEY` for LLM arbitration and reasoning steps.

---

## Model Checkpoints

Model weights should be placed in the `models/` directory:

| Path | Description | Download Link |
|---|---|---|
| `models/clip/` | Fine-tuned Multimodal CLIP vision-language head | `[TODO: download link]` |
| `models/muril_classifier/` | Fine-tuned MuRIL Hindi/Marathi bilingual classifier | `[TODO: download link]` |
| `models/faketype_en/` | Base RoBERTa English fake-type classifier | `[TODO: download link]` |
| `models/faketype_en_continued/` | Continued fine-tuned RoBERTa English fake-type model | `[TODO: download link]` |
| `models/faketype_muril/` | Cross-lingual MuRIL fake-type classifier | `[TODO: download link]` |

---

## Demo & Offline Execution

To run a reliable offline demo without external network dependencies:
1. **Warm the cache** with network access enabled to ensure all demo passages and searches are stored locally:
   ```bash
   python scripts/warm_demo_cache.py
   ```
2. **Enable offline cache-only mode** (`CACHE_ONLY=1`) to prevent any live network requests and verify reproducible execution from disk:
   ```bash
   # Windows (PowerShell):
   $env:CACHE_ONLY="1"; streamlit run app.py
   # Linux / macOS:
   CACHE_ONLY=1 streamlit run app.py
   ```

---

## Execution & Reproduction Commands

### 1. Interactive Web Interface
Launch the full Streamlit dashboard:
```bash
streamlit run app.py
```

### 2. Regression Sanity Suite
Run the core fact-checking and relation consistency test suite:
```bash
python scripts/regression_day1.py
```

### 3. Evaluation & Ablation Scripts
Reproduce the benchmark evaluations and ablation studies:
- **Verifier Ablation (C0-C5)**:
  ```bash
  python scripts/eval_verifier_ablation.py
  ```
- **Symmetric Reliability Rerun (C0 & C5)**:
  ```bash
  python scripts/rerun_c0_c5.py
  ```
- **Pattern Set Retrieval & Evaluation**:
  ```bash
  python scripts/rerun_v9d.py
  ```
- **Fake-Type Classification (English Benchmark)**:
  ```bash
  python scripts/eval_faketype_en.py
  ```
- **Cross-Lingual Fake-Type Evaluation**:
  ```bash
  python scripts/eval_faketype.py
  ```

---

## Repository Layout

```
Fake-News-ML-Project/
├── app.py                     # Streamlit web application entrypoint
├── requirements.txt           # Pinned python dependencies
├── .env.example               # Template environment configuration
├── src/                       # Core system modules
│   ├── fact_check/            # End-to-end fact checker and aggregation
│   ├── verification/          # Relational consistency & entity checkers
│   ├── reasoning/             # Normalizer and Groq LLM arbitration
│   ├── retrieval/             # DuckDuckGo & Wikipedia search and scraping
│   └── multimodal/            # CLIP vision-language and OCR handlers
├── scripts/                   # Evaluation, regression, and diagnostic scripts
├── data/                      # Evaluation benchmarks (git-ignored)
├── models/                    # Model weights and checkpoints (git-ignored)
├── results/                   # Evaluation tables, metrics, and logs
└── cache/                     # Disk cache for searches and pages (git-ignored)
```

---

## Key Experimental Results

### 1. Fact Verifier Ablation (REPORT Set, N=37)
Evaluated across six configurations on `data/eval/claims.csv` (excluding calibration rows T01-T06, F01-F06; source: `results/final_verifier_tables.md` and `results/verifier_rerun_c0_c5.csv`):

| Configuration | Accuracy | 95% Wilson CI | Macro-F1 | % Abstentions |
|---|---|---|---|---|
| C0: closed-book LLM | 27/37 (73.0%) | [57.0%, 84.6%] | 0.7374 | 27.0% (10/37) |
| C1: NLI only | 21/37 (56.8%) | [40.9%, 71.3%] | 0.5640 | 29.7% (11/37) |
| C2: NLI + consistency checker | 20/37 (54.1%) | [38.4%, 69.0%] | 0.5036 | 70.3% (26/37) |
| C3: C2 + competing-fact pass | 20/37 (54.1%) | [38.4%, 69.0%] | 0.5036 | 70.3% (26/37) |
| C4: full system (as run) | 31/37 (83.8%) | [68.9%, 92.3%] | 0.8378 | 35.1% (13/37) |
| **C4: full system (after reliability rerun)** | **33/37 (89.2%)** | **[75.3%, 95.7%]** | **0.8942** | **29.7% (11/37)** |
| **C5: LLM reading the same evidence (after reliability rerun)** | **34/37 (91.9%)** | **[78.7%, 97.2%]** | **0.9198** | **8.1% (3/37)** |

- **Rule Layer vs. Direct LLM Reading**: Comparing C4 (after reliability rerun: 33/37) against C5 (after reliability rerun: 34/37), the paired sign test shows only C4 right = 2, only C5 right = 3, both right = 31, both wrong = 1 ($p = 1.0000$). The rule layer did not add measured accuracy over C5.
- **Statistical Significance vs. Baseline**: C4 achieves significant gains over the pure NLI pipeline C1 ($p = 0.0005$, 12 C4-only wins vs. 0 C1-only wins), and outperforms closed-book C0 ($p = 0.1796$, 10 C4-only wins vs. 4 C0-only wins).
- **Relation-Pattern Diagnostics (exploratory, post hoc)**: Diagnostic analysis on relation claim patterns (`results/verifier_relation_pattern.csv`, N=5) showed that structural failures stemmed from zero-evidence retrieval misses rather than relation logic errors; with retry-enabled retrieval (`results/verifier_relation_pattern_rerun.csv`), 4/5 resolved correctly.

### 2. Fake-Type Classification
Evaluated across 4 classes (`ordinary`, `satire`, `clickbait`, `misleading`) across Dev, Test, Holdout, and the Pooled 100-row benchmark (source: `results/faketype_en_dev.csv`, `results/faketype_en_test.csv`, `results/faketype_en_holdout.csv`):

| Split | $n$ | Trained-only RoBERTa | LLM-only (LLaMA-3.3-70B) | Hybrid Pipeline (v1) |
|---|---|---|---|---|
| Dev | 28 | 16/28 (57.1%) | 24/28 (85.7%) | 25/28 (89.3%) |
| Test | 48 | 28/48 (58.3%) | 34/48 (70.8%) | 40/48 (83.3%) |
| Holdout | 24 | 15/24 (62.5%) | 17/24 (70.8%) | 17/24 (70.8%) |
| **Pooled Benchmark** | **100** | **59/100 (59.0%)**<br>[95% CI: 49.2%, 68.1%] | **75/100 (75.0%)**<br>[95% CI: 65.7%, 82.5%] | **82/100 (82.0%)**<br>[95% CI: 73.3%, 88.3%] |

---

## Known Limitations

1. **Live Search Fragility**: Upstream web search endpoints (HTML/Lite DuckDuckGo) can intermittently throttle requests or return 403s during rapid batch querying, necessitating fallback to Wikipedia and disk caching.
2. **Tense Blindness in Relational Extraction**: The relational consistency checker parses static subject-predicate-object tuples without temporal scoping, leading to false support on historical facts that have changed over time (e.g., R07p regarding Rio de Janeiro as Brazil's capital, or F07 regarding India's independence year).
3. **Weak NLI Contradiction Deciding Verdicts**: In cases with sparse evidence (e.g., N01 regarding the Hera spacecraft), a single weakly-scoring contradictory NLI passage can dominate if no countervailing support passages surpass thresholds.
4. **Narrow Rule Phrasing**: Dependency parse extraction relies on fixed predicate and relation mapping vocabularies, abstaining when assertions use unmapped synonyms or complex passive structures.
5. **No Cache-Based Reproducibility Beyond Stored Snapshot**: Full offline reproducibility is tied to the frozen disk cache snapshot recorded in **November 2026**; live web evolution may shift search ranking and passage snippets over time.

---

## License

This project is released under the [MIT License](LICENSE) (placeholder).
