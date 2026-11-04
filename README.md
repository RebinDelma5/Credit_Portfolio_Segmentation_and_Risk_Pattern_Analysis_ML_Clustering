# Credit Portfolio Segmentation and Risk Pattern Analysis

An unsupervised machine-learning project on the Lending Club loan data (1.35 million loans).
It groups borrowers with similar financial and credit characteristics into **segments**, then
checks how each segment repaid its loans in the past. Results are shown in an interactive
**Streamlit dashboard**.

> **Main rule of the project:** the `Default` column is **never** used to build the segments.
> It is used only afterwards to compare how each segment repaid.

---

## 1. Quick start

Python 3.12 was used for testing. A recent Python 3 version should also work.

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install the packages
pip install -r requirements.txt

# 3. start the dashboard (run this from the project folder)
streamlit run app.py
```

The dashboard works straight away. It reads files that are already in this folder
(`models/model.pkl` and the CSV files in `data/processed/`), so **you do not need the
dataset just to see the dashboard**.

### Rerun the whole analysis (optional)

1. Download `loans_granting_model_dataset.csv` and put it in `data/raw/`
   (it is 167 MB, so it is not included in the project or in Git).
2. Start Jupyter and open the notebook:
   ```bash
   jupyter lab
   ```
   Open `Credit_Portfolio_Segmentation.ipynb` and choose **Run > Run All Cells**.
   It takes about 5 minutes and needs about 3 GB of free memory.
3. The last cells of the notebook save the model to `models/model.pkl` (with `pickle`)
   and export the CSV files that the dashboard reads. Then run `streamlit run app.py` again.

---

## 2. What is in this folder

```
credit_segmentation/
├── README.md                              <- this file
├── requirements.txt                       <- packages to install
├── Credit_Portfolio_Segmentation.ipynb    <- the full analysis and the ML model (already run)
├── app.py                                 <- the Streamlit dashboard
├── models/
│   └── model.pkl                          <- trained model saved with pickle (used by app.py)
├── data/
│   ├── raw/                               <- put the dataset CSV here
│   └── processed/                         <- small CSV files that the dashboard reads
├── outputs/
│   ├── figures/                           <- EDA, clustering and map charts (PNG)
│   └── tables/                            <- result tables (CSV / JSON)
└── reports/
    ├── Project_Report.docx                <- full written report
    ├── Final_Presentation.pptx            <- simple 7-slide presentation
    └── Code_Explanation_Guide.pdf         <- explains every code cell and every app section
```

**Only one Python file:** `app.py` is the only `.py` file. All the analysis and the model are in the
notebook, which creates `models/model.pkl` and the CSV files in `data/processed/` that `app.py` reads.

---

## 3. Dataset

`loans_granting_model_dataset.csv` has 1,347,681 loans issued in 2007-2018 and 15 columns:
loan id, issue date, annual income (`revenue`), debt-to-income ratio (`dti_n`), loan amount,
FICO credit score (`fico_n`), employment length, loan purpose, home ownership, state, zip code,
`Default` (0 = repaid, 1 = defaulted) and three text/helper columns (`title`, `desc`,
`experience_c`).

---

## 4. Method in short

| Step | What was done |
|---|---|
| Clean | Dropped `id`, `desc` (91% empty), `title`, `zip_code`, `experience_c` (constant). Removed 532 rows with impossible DTI (999) and 517 rows with home ownership `OTHER`. Employment `NI` (no information, 77,906 rows) kept and filled with the median. Extreme income, DTI and loan-to-income values capped at the 99.5th percentile. |
| Features | log(1 + income), DTI, loan amount, FICO, employment years, loan-to-income, purpose (4 groups), home ownership. Numbers scaled with `StandardScaler`, categories one-hot encoded. |
| PCA | 7 components keep 94.0% of the information. |
| Cluster | K-Means (k = 5), Hierarchical (Ward) and DBSCAN compared. K-Means used all borrowers; the slow methods used a random sample of 10,000. |
| Visualise | PCA, t-SNE and UMAP maps. |
| Compare defaults | Default rate per segment with 95% confidence interval, chi-square test, by year, region and purpose. |
| Anomalies | DBSCAN noise points plus feature z-scores. |
| Save | Model saved with `pickle` to `models/model.pkl`. |

Random seed is `42` everywhere, so results are repeatable.

---

## 5. Results

K-Means found five segments among 1,346,632 cleaned borrowers:

| Segment | Share | Median FICO | Median income | Median loan | Default rate |
|---|---|---|---|---|---|
| 0 Established Homeowners | 23.9% | 682 | $72,000 | $9,600 | 16.7% |
| 1 Early-Career Renters | 23.4% | 682 | $58,000 | $8,000 | 18.1% |
| 2 Stretched Low-Income | 21.7% | 687 | $43,000 | $14,175 | **29.1%** |
| 3 Large-Loan High-Income | 20.7% | 697 | $100,000 | $26,400 | 22.1% |
| 4 Prime Credit | 10.3% | 762 | $75,000 | $12,000 | **8.4%** |

* The average default rate of all loans is 20.0%. The riskiest segment defaults about 3.5 times
  as often as the safest.
* Prime Credit has the lowest default rate in every issue year from 2008 to 2018. Stretched
  Low-Income has the highest in every year from 2010 to 2018.
* Affordability matters as much as credit score: Stretched Low-Income borrowers have an
  ordinary FICO score but the highest debt-to-income and loan-to-income ratios.

| Algorithm | Clusters | Silhouette score | Comment |
|---|---|---|---|
| K-Means | 5 | 0.145 | best scores and most interpretable: final model |
| Hierarchical (Ward) | 5 | 0.068 | weaker separation (agreement with K-Means: ARI 0.24) |
| DBSCAN | 1 + 2.2% noise | not available | data is one continuous cloud; used to find anomalies |

**Anomalies:** 219 of the 10,000 sampled borrowers (2.2%) were flagged. They are mostly
unusually high FICO scores or incomes, and their default rate (17.8%) is not higher than
normal (20.2%). They are unusual, not necessarily risky.

---

## 6. The dashboard (`app.py`)

Six tabs, all charts drawn with matplotlib and seaborn:

| Tab | What you can do |
|---|---|
| Overview | See the key numbers, the share of each segment and the algorithm comparison |
| Segments | Pick a segment and see its profile; compare segments; see the spread of any feature |
| Default rates | See default rate by segment (with confidence intervals) and by year |
| Map | View the borrowers in PCA, t-SNE or UMAP, coloured by segment or by repaid/default |
| Try a borrower | Enter a profile and see the closest segment (uses the pickled model) |
| Insights | Read the findings, recommendations and limitations |

---

## 7. Limitations

* The silhouette score is low (about 0.15). Segments overlap, so they are tendencies, not hard borders.
  k = 5 was chosen using the metrics **and** how meaningful the segments are.
* Loans from 2017-2018 have not had time to default, so their default rates look lower than they should.
* Only 8 borrower variables were used (no payment history or economic data).
* Hierarchical clustering, DBSCAN, t-SNE and UMAP used a 10,000-borrower sample.
* The segments describe past groups. They are **not** a credit-scoring model.

## 8. Ethical use

Segments describe **groups**. Never use a person's segment alone to approve or reject them.
No personal identifiers (loan id, zip code) are used for clustering or shown in the dashboard.
The "Try a borrower" tab is a learning demo and shows group history, not a prediction.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` when starting the app | Run `streamlit run app.py` from inside the project folder (the one that contains `models/` and `data/`). |
| `FileNotFoundError` for the CSV in the notebook | Put `loans_granting_model_dataset.csv` in `data/raw/` and start Jupyter from the project folder. |
| Error when loading `model.pkl` | Pickle files depend on the scikit-learn version. Install the versions in `requirements.txt` or rerun the notebook to create a new `model.pkl`. |
| Notebook stops or the computer freezes | The full data needs about 3 GB of free memory. Close other programs and run again. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again in the same environment you use for Jupyter / Streamlit. |

---

## 10. Put the project on GitHub

```bash
git init
git add .
git commit -m "Credit portfolio segmentation project"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

The included `.gitignore` leaves out the 167 MB dataset and other big temporary files. The
small placeholder file in `data/raw/` keeps that folder in your repository.

---

## 11. Technologies

Python, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, UMAP, Streamlit, Jupyter, pickle.

*Project brief: Credit Portfolio Segmentation and Risk Pattern Analysis (GUVI x HCL).*
