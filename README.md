# SI-TG2 — Comparison of Categorization Algorithms

This repository contains the group project for the SI course — a comparative study of supervised classification algorithms that categorize vehicles by country of origin (Europe, Japan, USA). The full report (Portuguese, IEEE format) and figures are in `LaTeX Report/`. The dataset used for the study is located in `/Dataset`.

Report
- LaTeX source: `LaTeX Report/Relatório.tex`  
  https://github.com/MartinhoCaeiro/SI-TG2/blob/main/LaTeX%20Report/Relat%C3%B3rio.tex  
- PDF: `LaTeX Report/Relatório.pdf`  
  https://github.com/MartinhoCaeiro/SI-TG2/blob/main/LaTeX%20Report/Relat%C3%B3rio.pdf

---

## About

This project compares four supervised learning algorithms (Decision Tree, Random Forest, Logistic Regression and Neural Network) on a vehicle information dataset (Kaggle) to predict the vehicle's country of origin. Experiments were performed in Orange (GUI) with preprocessing (normalization, categorical encoding) and 10‑fold cross validation. Evaluation metrics include AUC, classification accuracy (CA), F1, precision, recall and MCC.

---

## Dataset

- Location in the repository: `/Dataset`  
- Source: "Car information dataset" (Kaggle) — the dataset used is a processed version from the course's earlier assignment.
- Typical attributes: make/model, fuel economy, number of cylinders, displacement, horsepower, weight, acceleration, year (year was discarded for the study), country of origin.
- Rows: ~400 (processed version used in the study).
- Preprocessing applied (as described in the report):
  - Removed the `year` column;
  - Normalized numeric attributes;
  - Encoded categorical attributes (e.g., brand);
  - Evaluation: 10‑fold cross validation.

If CSV files are not present in `/Dataset`, download the original dataset from Kaggle and apply the same preprocessing (I can add a preprocessing script if you want).

---

## Results (summary from the report)

Aggregated comparison table (values reported in the paper):

| Algorithm             | AUC   | CA    | F1    | Precision | Recall | MCC   |
|-----------------------|-------|-------|-------|-----------|--------|-------|
| Tree                  | 0.833 | 0.789 | 0.792 | 0.798     | 0.789  | 0.613 |
| Logistic Regression   | 0.909 | 0.775 | 0.777 | 0.783     | 0.775  | 0.587 |
| Random Forest         | 0.921 | 0.775 | 0.776 | 0.777     | 0.775  | 0.584 |
| Neural Network        | 0.900 | 0.753 | 0.753 | 0.756     | 0.753  | 0.544 |

Notes from the report:
- Random Forest achieved the highest AUC.
- The Decision Tree showed the highest classification accuracy in this study.
- Logistic Regression performed well in F1 and MCC.
- The Neural Network produced the worst performance here, possibly due to the small dataset size.

See the full report for per-algorithm confusion matrices and ROC curves (figures are in `LaTeX Report/Resources`).

---

## Reproducing the experiments — Orange (GUI)

The original experiments were executed with Orange (data mining GUI). To reproduce the Canvas workflow:

Prerequisites
- Orange (use the app or install via pip/conda)
  - pip: `pip install orange3`
  - conda: `conda install -c conda-forge orange3`
- Python 3.8+ (if using Orange installed via pip)
- (Optional) scikit-learn, pandas, matplotlib if you prefer scripted reproduction

Orange Canvas workflow
1. Start Orange Canvas: `orange-canvas` (or `python -m Orange.canvas`).
2. File widget → load the file(s) from `/Dataset`.
3. Preprocess → apply Normalization and encode categorical features.
4. Add learners: Decision Tree, Random Forest, Logistic Regression, Neural Network (MLP).
5. Test & Score:
   - Evaluation: Cross Validation
   - Folds: 10
6. Connect Confusion Matrix and ROC widgets to inspect per-class results.
7. Export results from Test & Score (Save as file) for comparison with the report.

---

## Reproducing the experiments — scikit-learn (example)

If you prefer a code-based reproduction, use scikit-learn. Minimal example pipeline (Python):

```python
# examples/reproduce_with_sklearn.py
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# 1. Load data (adjust filename)
df = pd.read_csv("Dataset/car_information_processed.csv")

# 2. Features / target
X = df.drop(columns=["country", "year"], errors='ignore')  # year was removed in the study
y = df["country"]

# 3. Preprocessing
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])

# 4. Models to compare
models = {
    "Tree": DecisionTreeClassifier(random_state=0),
    "RandomForest": RandomForestClassifier(random_state=0, n_estimators=100),
    "LogisticRegression": LogisticRegression(max_iter=1000, multi_class="auto"),
    "NeuralNetwork": MLPClassifier(max_iter=500, random_state=0),
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]  # extend as needed

for name, clf in models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False)
    print(name, {k: v.mean() for k, v in res.items() if k.startswith("test_")})
```

Notes:
- Compute multiclass AUC (one-vs-rest) separately if needed.
- For MCC, use `sklearn.metrics.matthews_corrcoef` with `cross_val_predict`.

---

## Project structure (current / suggested)

- LaTeX Report/  
  - Relatório.tex  
  - Relatório.pdf  
  - Resources/ (ROC images: `ROC_EUA.png`, `ROC_EU.png`, `ROC_JP.png`)  
- Dataset/  
  - (CSV(s) used — placed here; e.g., `car_information_processed.csv`)  
- examples/ (optional)  
  - reproduce_with_sklearn.py (example shown above)  
- README.md — this file
- LICENSE — (add if applicable)

I can add a notebook `notebooks/reproduce.ipynb` with step-by-step reproduction if you want.

---

## Contributing

1. Fork the repository.
2. Add reproducible scripts or notebooks (place in `examples/` or `notebooks/`).
3. If you add data, place files in `/Dataset` (verify dataset license) or add `Dataset/README.md` with Kaggle download instructions.
4. Open a Pull Request describing your changes.

---

## Authors / Contact

- Martinho Caeiro — 23917 — 23917@stu.ipbeja.pt  
- Paulo Abade — 23919 — 23919@stu.ipbeja.pt  
Instituto Politécnico de Beja — Escola Superior de Tecnologia e Gestão

---

## License

- This repository is licensed under the GNU General Public License v3.0 (GPL-3.0).
