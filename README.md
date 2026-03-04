# A Comparison between SVMs and Logistic Regression

In this project, we compare the performance of support vector machines (SVMs) and logistic regression in a classification setting where the true decision boundary is non-linear and the observations are well separated.

## Project Structure 

- `src/` - source code for models and evaluations;
- `report_and_results/` - detailed report of the results obtained;

## Dataset

The dataset consists of 500 observations. There are two predictors, `x1` and `x2`, generated i.i.d from a standard uniform distribution, with a fixed seed, shifted by `-0.5`. The response variable, `y` has two levels: `True` if `x1^2 - x2^2 > 0` and `False` otherwise.

# Results Summary 

| Model | Predictors / Kernel | Tuning | Accuracy |
|---|---|---|---:|
| Logistic Regression | x1, x2 | — | 43.0% |
| Logistic Regression | x1, x2, x1^2, x2^2 | — | 95.4% |
| Logistic Regression | x1, x2, x1x2, x1^2, x2^2 | — | 95.8% |
| SVM (linear) | linear kernel | C = 1 (CV over C had no meaningful effect) | 50.6% |
| SVM (poly) | polynomial kernel (degree 2) | C = 1 | 98.0% |

It is with no surprise that the degree-2 polynomial SVM achieves the best results. Although we also tested logistic regression with polynomial predictors, the decision rule is explicitly defined and yields a clear nonlinear separation between classes. In such well-separated settings, SVMs often outperform logistic regression.

## How to run 

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

### 2) Install Dependencies
```bash 
pip install -r requirements.txt
```

### 3) Run the Project 
```bash
python src/main.py
```

