# Exploratory Data Analysis & Machine Learning on Unstructured Productivity, Task-Management, and Psychological Stress Text

**Faculty Guide:** Mrs. Nalini N  
**Contributors:**  
- Sharvaree Harkare (25BDS0217)  
- Disha Chandra Mani (25BDS0172)  

---

## 📌 Project Overview & Theoretical Background
Academic procrastination, executive dysfunction, and chronic task postponement are intimately entangled with emotional distress, anxiety, and depressive symptoms. Grounded in a survey of **15 academic papers**, this project demonstrates that procrastination is fundamentally an **emotional regulation failure under acute evaluation pressure** (Pychyl & Sirois, 2016) rather than mere poor scheduling.

When students or knowledge workers face impending deadlines, cognitive overload triggers measurable linguistic shifts:
- **Temporal Distortion:** Excessive past rumination (`focuspast`) and future dread/avoidance.
- **Egocentric Narrowing:** Surges in first-person singular pronouns (`lex_liwc_i` / *I*-talk).
- **Cognitive Discrepancy:** Elevated modal conflict words (`should`, `would`, `could`).
- **Emotional Valence Collapse:** Severe drop in emotional tone (`lex_liwc_Tone`).

This repository provides an end-to-end data science investigation spanning **preprocessing, exploratory visual profiling, non-parametric hypothesis testing, unsupervised topic modeling (LDA), and supervised predictive classification** across two rich social media corpora:
1. **The Dreaddit Stress Corpus** ($N = 3,529$ annotated posts across 10 acute distress subreddits).
2. **The Reddit ADHD Community Corpus** ($N = 4,933$ curated submissions on executive dysfunction, deadline panic, and task-initiation coping).

---

## 📂 Repository Structure

```text
├── eda_project.ipynb                   # Master 51-cell interactive notebook (executed with live outputs)
├── Explanation.docx                    # 54 KB Master Technical Reference & 20-Question Viva Voce Cheat Sheet
├── EDA PROJECT (1).pdf                 # Comprehensive Literature Review (15 papers surveyed)
├── figures/                            # 11 publication-grade visualizations + CSV benchmarks
│   ├── fig1_subreddit_stress_distribution.png
│   ├── fig2_text_complexity_readability.png
│   ├── fig3_affect_emotion_distribution.png
│   ├── fig4_temporal_focus_dynamics.png
│   ├── fig5_cognitive_discrepancy.png
│   ├── fig6_psychological_correlation_matrix.png
│   ├── fig7_top_distinguishing_ngrams.png
│   ├── fig8_adhd_lda_topics.png
│   ├── fig9_model_roc_curves.png
│   ├── fig10_confusion_matrices.png
│   ├── fig11_top_predictive_features.png
│   ├── model_comparison_benchmark.csv
│   └── statistical_hypothesis_results.csv
├── data/
│   ├── dreaddit/                       # Raw Dreaddit train/test splits
│   ├── dreaddit_processed_enriched.csv # 3,529 posts × 51 psychological & engineered features
│   └── adhd_processed_sample.csv       # 4,933 curated ADHD posts with NLP metrics & LDA topics
├── .gitignore                          # Excludes raw multi-gigabyte files (>100MB)
└── README.md                           # Project documentation
```

---

## 🔬 Key Methodological Innovations

1. **Memory-Safe ADHD Streaming:** Replaced bulk loading of a 1.24 GB raw comment dump with a streaming chunk iterator (`chunksize=10,000`), curating 4,933 representative submissions while consuming $<15\text{ MB}$ of RAM.
2. **Contraction Expansion Pipeline:** Implemented a 40+ pattern expansion dictionary (`"I'm"` $\to$ `"I am"`, `"don't"` $\to$ `"do not"`), eliminating vocabulary corruption (`im`, `dont`) and preserving negation markers crucial for sentiment detection.
3. **POS-Aware WordNet Lemmatization:** Mapped Penn Treebank tags to WordNet POS categories (`VERB`, `ADJ`, `NOUN`, `ADV`), correctly reducing action verbs (*procrastinating* $\to$ *procrastinate*, *avoiding* $\to$ *avoid*).
4. **Behavioral Surface Metrics:** Engineered structural proxies for vocal urgency and cognitive looping: Type-Token Ratio (TTR), uppercase ratio, sentence count, and emotional punctuation (question marks for reassurance-seeking, exclamation marks for panic).

---

## 📊 Inferential Statistical Hypothesis Testing

Two-sample **Mann-Whitney U non-parametric tests** and **Cohen's $d$ effect sizes** comparing Stressed ($N=1,848$) vs. Non-Stressed ($N=1,681$) narratives:

| Feature | Stressed Mean | Non-Stressed Mean | $p$-value | Cohen's $d$ | Effect Magnitude | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`lex_liwc_Tone`** | 18.04 | 49.50 | $< 10^{-160}$ | **-1.0031** | **Large** | Collapse of positive emotional tone under stress. |
| **`lex_liwc_i`** | 10.84% | 7.31% | $< 10^{-114}$ | **+0.8316** | **Large** | Egocentric narrowing (*I*-talk spike). |
| **`sentiment`** | -0.016 | +0.105 | $< 10^{-85}$ | **-0.6526** | **Medium** | Significant negative valence shift. |
| **`lex_liwc_anx`** | 1.31% | 0.51% | $< 10^{-72}$ | **+0.5619** | **Medium** | $2.5\times$ surge in acute anxiety and panic words. |
| **`lex_liwc_anger`** | 1.22% | 0.60% | $< 10^{-47}$ | **+0.4446** | **Small-Med** | Frustration and irritability during crisis. |
| **`lex_liwc_discrep`**| 1.89% | 1.42% | $< 10^{-18}$ | **+0.2814** | **Small** | Procrastination guilt marker (*should, would, could*). |
| **`lex_liwc_focuspast`**| 4.82% | 4.11% | $< 10^{-12}$ | **+0.2205** | **Small** | Backward-looking depressive rumination. |
| **`syntax_fk_grade`**| 5.82 | 5.88 | $0.412$ | **-0.0284** | **Negligible** | Stress does not impair grammatical reading level. |

---

## 🧠 Unsupervised Topic Modeling: Latent Dirichlet Allocation (LDA)

Fitted an LDA topic model ($K=5$) on 4,933 ADHD submissions:
- **Topic 1:** *Academic Deadlines & School Challenges* (`school`, `college`, `class`, `fail`, `work`)
- **Topic 2:** *Daily Medication Management & Cognitive Focus* (`adderall`, `med`, `doctor`, `effect`, `focus`)
- **Topic 3:** *Task Initiation, Habit Building & Procrastination Coping* (~20% prevalence; `task`, `list`, `start`, `clean`, `goal`)
- **Topic 4:** *Emotional Dysregulation & Burnout* (`feel`, `tired`, `cry`, `overwhelm`, `struggle`)
- **Topic 5:** *Community Support & Peer Accountability* (`win`, `post`, `share`, `tip`, `wednesday`, `advice`)

---

## 🤖 Supervised Machine Learning Benchmark

Benchmarked three classifiers on a multi-modal feature space combining **1,000 TF-IDF n-grams** and **27 standardized LIWC & surface features** (1,027 total dimensions) using a stratified 80/20 train/test split:

| Model | Accuracy (%) | Precision (%) | Recall (Sensitivity) (%) | F1-Score (%) | ROC-AUC | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression ($L_2$)** | **75.21%** | **75.19%** | **78.65%** | **76.88%** | **0.8490** | **0.06s** |
| **Linear SVC (Calibrated)** | 74.50% | 73.63% | 80.00% | 76.68% | 0.8398 | 0.49s |
| **Random Forest (Ensemble)**| 73.94% | 72.14% | **81.89%** | 76.71% | 0.8281 | 0.41s |

### Model Explainability & Key Drivers:
- **Strongest Predictors of Stress:** First-person singular pronouns (`lex_liwc_i`: $+0.875$), anxiety markers (`lex_liwc_anx`: $+0.303$), present-moment urgency (`lex_liwc_focuspresent`: $+0.341$), and crisis tokens (`hard`: $+1.61$, `job`: $+1.51$, `feel like`: $+1.05$).
- **Strongest Protective Indicators:** Emotional tone (`lex_liwc_Tone`: $-0.431$), pleasantness ($-0.315$), and agency tokens (`avoid`: $-1.78$, `finally`: $-1.28$, `thank`: $-1.07$).

---

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DCMani2006/Exploratory-Data-Analysis-of-Unstructured-Productivity-and-Task-Management-Text.git
   cd Exploratory-Data-Analysis-of-Unstructured-Productivity-and-Task-Management-Text
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn nltk scipy scikit-learn python-docx
   ```

3. **Launch the master notebook:**
   ```bash
   jupyter notebook eda_project.ipynb
   ```
   *All 51 cells execute sequentially with pre-rendered figures, statistical summaries, and classification reports.*
