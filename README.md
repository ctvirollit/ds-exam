# 1‑Hour Data Science Exam

**Title:** Random Forest Entity Matching (Lite)

**Time Limit:** 60 minutes (hard stop)

**Goal:** Train a simple `RandomForestClassifier` that predicts whether two records refer to the **same person**.

---

## Context

You will receive a small, synthetic dataset of record **pairs** (A vs B). Each row has an `is_match` label (1 = same person, 0 = different). There are mild typos and nickname variations to simulate real‑world data quality.

---

## Files Provided

* `pairs_small.csv` – \~120 labeled record pairs
* `starter.py` – a minimal scaffold: load → **featurize()** → train → evaluate

> You only need to **complete `featurize()`** and keep the modeling simple.

---

## Environment Setup (suggested)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -U pip
pip install pandas scikit-learn   # optional extras:
pip install rapidfuzz jellyfish   # (string similarity, phonetics)
python starter.py
```

Allowed packages: **pandas, scikit‑learn**. Optional (nice to have, not required): **rapidfuzz, jellyfish**.

Not allowed: AutoML frameworks, deep learning models, external datasets.

---

## Task Breakdown

1. **Scan the data (5 min).**

   * Columns come in A/B pairs: `a_first_name, a_last_name, a_dob, a_city, a_mobile` and the corresponding `b_*` columns, plus `is_match`.

2. **Feature Engineering (20–25 min).**
   Implement **4–8 features** inside `featurize(df)` (in `starter.py`). Keep it fast and simple:

   * Exact/near‑exact flags: last name equality, DOB equality, city equality.
   * Simple string similarity for first name (e.g., character‑set overlap already scaffolded; you may replace with `rapidfuzz` if installed).
   * Phone last‑4 match indicator.
   * (Optional) Any other quick, leakage‑free signal you can compute.

3. **Modeling & Evaluation (20–25 min).**

   * Use the scaffolded **train/test split** (75/25, `random_state=42`, stratified).
   * Train a `RandomForestClassifier` (sensible defaults; `class_weight="balanced"` is fine).
   * Print **confusion matrix** and **classification report** (precision/recall/F1).
   * Optional: adjust decision threshold (explain in 1–2 sentences if you do).

4. **Short Write‑Up (≤5 min).**

   * 3–5 bullets: which features you created, any threshold choice, quick notes on false positives/negatives.

---

## What to Submit

**Within 60 minutes**, send a zip/folder containing:

* `starter.py` (with your completed `featurize()`; additional helper functions OK)
* `README.md` (≤1 page) with:

  * Brief feature list & rationale (bullets)
  * Metrics on the holdout set (precision, recall, F1)
  * Notes on thresholding (if any) and 1–2 observations about errors
* (Optional) `requirements.txt` **if** you used optional libraries

> Do **not** submit large artifacts or notebooks. Keep it lightweight.

---

## Grading Rubric (100 pts)

* **Metrics (70 pts)**

  * F1 score (40)
  * Recall (20)
  * Precision (10)
* **Features (20 pts)** – 4–8 relevant, leakage‑free features
* **Hygiene (10 pts)** – clear code, reproducible (`random_state`), concise README

**Passing guide:** F1 ≥ **0.85** on the provided split is achievable with simple features.

---

## Rules & Guardrails

* **No leakage.** Don’t derive features from the label or from any ID that trivially reveals the answer.
* Keep it **fast**. No heavy hyper‑parameter tuning or external data.
* You may consult documentation, but all code must be **your own**.

---

## Tips

* Start with exact/equality features (they’re strong baselines).
* Add one string‑similarity feature (first name) and one contact/location cue.
* If classes are imbalanced, `class_weight="balanced"` helps.
* If precision vs recall trade‑off matters, mention your chosen threshold.

---

## Expected Console Output (format)

Your script should print something like:

```
[[TN  FP]
 [FN  TP]]
              precision    recall  f1-score   support

           0      ...        ...      ...        ...
           1      ...        ...      ...        ...

    accuracy                          ...        ...
   macro avg      ...        ...      ...        ...
weighted avg      ...        ...      ...        ...
```

---

## FAQ

* **Can I add more features?** Yes, but stay within 4–8 to fit the timebox.
* **Can I change model type?** Prefer Random Forest. If you try another simple classifier, justify briefly.
* **Can I use `rapidfuzz`/`jellyfish`?** Optional. If not installed, keep your similarity functions simple.

Good luck!
