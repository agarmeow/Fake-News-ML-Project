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
│   ├── retrieval/             # DuckDuckGo search and web scraper
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
Evaluated across six configurations on `data/eval/claims.csv` (excluding calibration rows T01-T06, F01-F06; source: `results/final_verifier_tables.md`):

| Configuration | Accuracy | 95% Wilson CI | Macro-F1 | % Abstentions |
|---|---|---|---|---|
| C0: Closed-book LLM (no retrieval) | 27/37 (73.0%) | [57.0%, 84.6%] | 0.7374 | 27.0% (10/37) |
| C1: NLI only | 21/37 (56.8%) | [40.9%, 71.3%] | 0.5640 | 29.7% (11/37) |
| C2: NLI + Consistency checker | 20/37 (54.1%) | [38.4%, 69.0%] | 0.5036 | 70.3% (26/37) |
| C3: C2 + Competing-fact search | 20/37 (54.1%) | [38.4%, 69.0%] | 0.5036 | 70.3% (26/37) |
| C4 (as run): Full system | 31/37 (83.8%) | [68.9%, 92.3%] | 0.8378 | 35.1% (13/37) |
| **C4 (after reliability rerun): Full system** | **33/37 (89.2%)** | **[75.3%, 95.7%]** | **0.8942** | **29.7% (11/37)** |
| C5: LLM reading retrieved evidence | 31/37 (83.8%) | [68.9%, 92.3%] | 0.8394 | 40.5% (15/37) |

- **Statistical Significance**: Paired sign test for C4 (rerun) vs. C1 (NLI-only) achieves $p = 0.0005$ (12 C4-only wins vs. 0 C1-only wins). C4 (rerun) vs. C0 yields $p = 0.1796$.
- **High-Confidence Precision**: In the strong-evidence band (confidence $> 0.60$), C4 achieves 95.2% accuracy (20/21).

### 2. Fake-Type Classification (English Pooled Dev+Test, N=76)
Evaluated across 4 classes (`ordinary`, `satire`, `clickbait`, `misleading`; source: `results/faketype_en_final.txt`):

| Method | Accuracy | 95% Wilson CI | Macro-F1 |
|---|---|---|---|
| Trained-only RoBERTa | 57.89% (44/76) | [46.68%, 68.35%] | 0.5015 |
| LLM-only (LLaMA-3.3-70B) | 76.32% (58/76) | [65.64%, 84.46%] | 0.7711 |
| **Hybrid Pipeline** | **85.53% (65/76)** | **[75.91%, 91.72%]** | **0.8498** |

---

## Known Limitations

1. **External Search Engine Rate Limits**: Web retrieval depends on DuckDuckGo HTML endpoints, which may impose transient rate limits during unpaced batch queries if the local disk cache is disabled.
2. **Temporal Reasoning for Historical Relations**: Entity/relation consistency matches static predicate tuples without time scoping; historical facts (e.g. past national capitals) may match as true under relational matching, requiring LLM arbitration to resolve temporal qualification.
3. **Zero-Evidence Obscure Entities**: Obscure or synthetically invented claims with zero search results default to `UNVERIFIABLE`.
4. **Cross-Lingual Relational Depth**: Detailed semantic relation extraction is optimized for English grammatical dependencies, while Hindi and Marathi rely primarily on MuRIL cross-lingual representations and translation-assisted NLI.

---

## License

This project is released under the [MIT License](LICENSE) (placeholder).
