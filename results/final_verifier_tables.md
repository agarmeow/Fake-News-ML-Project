# Final Verifier Evaluation Tables

### 1. REPORT Set Performance (N=37)

| Configuration | Accuracy | 95% Wilson CI | Macro-F1 | % UNVERIFIABLE (Abstentions) |
|---|---|---|---|---|
| C0 (Direct Search + Per-Source NLI) | 27/37 (73.0%) | [57.0%, 84.6%] | 0.7374 | 27.0% (10/37) |
| C1 (Source Tiering + Scraping) | 21/37 (56.8%) | [40.9%, 71.3%] | 0.5640 | 29.7% (11/37) |
| C2 (NLI + Relation/Entity Consistency) | 20/37 (54.1%) | [38.4%, 69.0%] | 0.5036 | 70.3% (26/37) |
| C3 (C2 + Competing-Fact Search) | 20/37 (54.1%) | [38.4%, 69.0%] | 0.5036 | 70.3% (26/37) |
| C4 (as run) | 31/37 (83.8%) | [68.9%, 92.3%] | 0.8378 | 35.1% (13/37) |
| **C4 (after reliability rerun)** | **33/37 (89.2%)** | **[75.3%, 95.7%]** | **0.8942** | **29.7% (11/37)** |
| C5 (LLM Direct Evidence Reading) | 31/37 (83.8%) | [68.9%, 92.3%] | 0.8394 | 40.5% (15/37) |

---

### 2. Paired Comparisons with Rerun-Corrected C4 (N=37)

| Comparison | Only C4 Right | Only Other Right | Both Right | Both Wrong | Discordant ($n_d$) | Exact Sign-Test $p$-value |
|---|---|---|---|---|---|---|
| **C4 (rerun) vs C0** | 10 | 4 | 23 | 0 | 14 | $p = 0.1796$ |
| **C4 (rerun) vs C1** | 12 | 0 | 21 | 4 | 12 | $p = 0.0005$ |
| **C4 (rerun) vs C5** | 2 | 0 | 31 | 4 | 2 | $p = 0.5000$ |

---

### 3. Accuracy by Subgroup

#### 3A. By Category (REPORT Set, N=37)

| Category | $n$ | C0 Accuracy | C4 (after rerun) Accuracy | C5 Accuracy |
|---|---|---|---|---|
| stable_fact | 8 | 8/8 (100.0%) | 8/8 (100.0%) | 8/8 (100.0%) |
| wrong_fact | 9 | 9/9 (100.0%) | 7/9 (77.8%) | 7/9 (77.8%) |
| invented | 12 | 8/12 (66.7%) | 11/12 (91.7%) | 11/12 (91.7%) |
| recent_news | 4 | 1/4 (25.0%) | 3/4 (75.0%) | 1/4 (25.0%) |
| recent_news_mutated | 4 | 1/4 (25.0%) | 4/4 (100.0%) | 4/4 (100.0%) |

#### 3B. By Language Group (REPORT Set, N=37)

| Language Group | $n$ | C0 Accuracy | C4 (after rerun) Accuracy | C5 Accuracy |
|---|---|---|---|---|
| English (`en`) | 32 | 22/32 (68.8%) | 28/32 (87.5%) | 26/32 (81.2%) |
| Non-English (`non-en`) | 5 | 5/5 (100.0%) | 5/5 (100.0%) | 5/5 (100.0%) |

---

### 4. C4 (after rerun) by Evidence-Strength Band
*(exploratory, thresholds fixed before these results)*

| Band | Confidence Range | $n$ | Accuracy | Decisive-Wrong Verdicts |
|---|---|---|---|---|
| Weak | $< 0.30$ | 12 | 11/12 (91.7%) | 1 (N01: pred FALSE, gold TRUE, conf=0.26) |
| Moderate | $0.30 - 0.60$ | 4 | 2/4 (50.0%) | 2 (F07: conf=0.48; F11: conf=0.41) |
| Strong | $> 0.60$ | 21 | 20/21 (95.2%) | 1 (U10: pred FALSE, gold UNVERIFIABLE, conf=0.85) |

---

### 5. Relation Pattern Abstentions (`results/verifier_relation_pattern.csv`, N=5)

| ID | Gold | C4 Verdict | Verdict Source | LLM Status | LLM Verdict | LLM Reasoning |
|---|---|---|---|---|---|---|
| R03p | TRUE | UNVERIFIABLE | rule_based | ok | none | n/a (search returned 0 evidence passages; LLM arbitration not invoked) |
| R04p | TRUE | UNVERIFIABLE | rule_based | ok | none | n/a (search returned 0 evidence passages; LLM arbitration not invoked) |
| R05p | TRUE | UNVERIFIABLE | rule_based | ok | none | n/a (search returned 0 evidence passages; LLM arbitration not invoked) |
| R06p | TRUE | UNVERIFIABLE | rule_based | ok | none | n/a (search returned 0 evidence passages; LLM arbitration not invoked) |
| R07p | FALSE | UNVERIFIABLE | rule_based | ok | none | n/a (search returned 0 evidence passages; LLM arbitration not invoked) |

---

### 6. Remaining C4 Errors After Rerun (REPORT Set, N=4)

| ID | Gold | C4 Verdict | Verdict Source | Confidence | Conflict Flag | Claim |
|---|---|---|---|---|---|---|
| F07 | FALSE | TRUE | rule_based | 0.48 | True | India gained independence in 1950. |
| F11 | FALSE | TRUE | rule_based | 0.41 | True | Vaccines cause autism. |
| U10 | UNVERIFIABLE | FALSE | llm_fallback | 0.85 | False | Engineer Rohit Valunjkar patented a self-folding bicycle in Thane in 2018. |
| N01 | TRUE | FALSE | rule_based | 0.26 | False | Europe's Hera spacecraft has begun close inspections of an asteroid rammed by NASA in a 2022 planetary defense test. |
