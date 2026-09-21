# Complete Project Walkthrough: EDA, Machine Learning & Viva Voce Guide

## Project Status: Complete End-to-End Execution

The project has now achieved full end-to-end implementation across all planned stages:
1. **Interactive Notebook:** [`eda_project.ipynb`](file:///c:/Users/bhuan/Desktop/EDA_Project/eda_project.ipynb) (51 modular cells, 6 cohesive parts, fully executed with zero errors).
2. **Master Defense Document:** [`Explanation.docx`](file:///c:/Users/bhuan/Desktop/EDA_Project/Explanation.docx) (54.4 KB formatted Word document covering every feature, 11 figure interpretations, statistical test results, ML benchmark, and a **20-Question Viva Voce Cheat Sheet**).
3. **Figure Suite:** 11 publication-quality plots archived in [`figures/`](file:///c:/Users/bhuan/Desktop/EDA_Project/figures).
4. **Data Assets:** Enriched datasets saved in [`data/`](file:///c:/Users/bhuan/Desktop/EDA_Project/data).

---

## 1. Machine Learning Performance Benchmark (Step 22)

We benchmarked three competitive classifiers on a multi-modal feature space combining **1,000 TF-IDF n-grams** and **27 standardized psycholinguistic & surface features** (1,027 total dimensions):

| Model | Accuracy (%) | Precision (%) | Recall (Sensitivity) (%) | F1-Score (%) | ROC-AUC | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (L2)** | **75.21%** | **75.19%** | **78.65%** | **76.88%** | **0.8490** | **0.06s** |
| **Linear SVC (Calibrated)** | 74.50% | 73.63% | 80.00% | 76.68% | 0.8398 | 0.49s |
| **Random Forest (Ensemble)**| 73.94% | 72.14% | **81.89%** | 76.71% | 0.8281 | 0.41s |

### Key Machine Learning Takeaways:
- **Logistic Regression is the Champion Model:** Achieves the highest discrimination capability (**ROC-AUC = 0.849**, $\approx 85\%$) and optimal precision/recall balance.
- **Why Linear Classifiers Excel Here:** In high-dimensional, sparse text spaces (1,027 dimensions), the decision boundary is approximately linear. Linear models thrive, whereas Random Forest struggles with orthogonal splits on sparse n-grams.
- **Recall Optimization for Mental Health Screening:** All models achieve high sensitivity ($78.6\% - 81.9\%$), minimizing False Negatives—vital in crisis surveillance so distressed individuals are not missed.

---

## 2. Model Explainability & Feature Importance (Step 23)

By inspecting the learned weights ($\beta$) of our Logistic Regression classifier, we identified the strongest mathematical drivers of stress:

### Top Drivers of Stress (Positive Log-Odds Weights):
1. **`lex_liwc_i` (+0.8750):** First-person singular pronouns (*I*-talk) is the single strongest continuous predictor of stress, confirming the clinical hypothesis of egocentric narrowing during panic.
2. **`lex_liwc_focuspresent` (+0.3414):** Present-tense visceral distress and urgency.
3. **`lex_liwc_anx` (+0.3030):** Acute anxiety and panic vocabulary.
4. **`hard` (+1.6108), `job` (+1.5057), `feel like` (+1.0489):** Colloquial indicators of perceived difficulty and burden.

### Top Protective Indicators (Negative Log-Odds Weights):
1. **`lex_liwc_Tone` (-0.4309):** High positive affective tone significantly decreases the odds of stress.
2. **`lex_dal_avg_pleasantness` (-0.3155):** Semantic pleasantness.
3. **`avoid` (-1.7772), `finally` (-1.2773), `thank` (-1.0691):** Words reflecting resolution, agency, and gratitude.

---

## 3. Out-of-Domain Transfer to ADHD Discourse (Step 24)

Deploying our trained classifier on 4,933 unlabelled ADHD submissions revealed:
- **50.5%** of overall ADHD discourse reflects acute psychological distress, while **49.5%** reflects constructive coping.
- Across our 5 LDA latent topics:
  - **Task Initiation & Procrastination Coping (Topic 3):** 54.4% acute stress prevalence.
  - **Community Support & Guidance (Topic 5):** 61.2% emotional expression / support seeking.
  - **Academic Deadlines & Medication (Topics 1 & 2):** Over 95% constructive discussion of treatment and logistics.

---

## 4. Complete File Manifest

| File / Folder | Details |
| :--- | :--- |
| [`eda_project.ipynb`](file:///c:/Users/bhuan/Desktop/EDA_Project/eda_project.ipynb) | 51 cells, fully executed with embedded charts, tables, and classification reports. |
| [`Explanation.docx`](file:///c:/Users/bhuan/Desktop/EDA_Project/Explanation.docx) | 54.4 KB master Viva Voce defense guide with a **20-Question Examiner Cheat Sheet**. |
| [`figures/`](file:///c:/Users/bhuan/Desktop/EDA_Project/figures) | 11 PNG plots + `model_comparison_benchmark.csv` + `statistical_hypothesis_results.csv`. |
| [`data/dreaddit_processed_enriched.csv`](file:///c:/Users/bhuan/Desktop/EDA_Project/data/dreaddit_processed_enriched.csv) | 3,529 posts $\times$ 51 features. |
| [`data/adhd_processed_sample.csv`](file:///c:/Users/bhuan/Desktop/EDA_Project/data/adhd_processed_sample.csv) | 4,933 ADHD submissions with engineered NLP metrics, LDA topics, and predicted stress probabilities. |
