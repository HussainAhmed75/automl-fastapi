# AutoML FastAPI

Upload a CSV (or Excel) file, pick the target column, and get back the best model for your data along with a full report.

I built this because I kept repeating the same steps on every new dataset: cleaning, encoding, scaling, trying a dozen models and comparing them. This API does all of that in one request.

## What it does

1. **Detects the task**: classification or regression, based on the target column.
2. **Cleans the data**: drops ID-like columns, constant columns and columns that are more than 80% empty, extracts features from date columns, and can optionally remove outliers (IQR).
3. **Profiles the dataset**: shape, missing values, duplicates, outliers and basic statistics.
4. **Preprocesses**:
   - Numeric: median imputation + `RobustScaler`
   - Low-cardinality categories: One-Hot Encoding
   - High-cardinality categories: Target Encoding
5. **Trains and compares 10+ models** with cross-validation: Logistic/Linear models, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, LightGBM, KNN, SVM, Naive Bayes (and CatBoost if installed).
6. **Tunes the best model** with Optuna (optional nested cross-validation).
7. **Returns the results** as a PDF report with charts, or as JSON.

## Run it locally

```bash
git clone https://github.com/HussainAhmed75/automl-fastapi.git
cd automl-fastapi
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/docs` to try it from the browser.

## API

`POST /automl` (multipart form)

| Field | Default | Description |
|---|---|---|
| `file` | required | CSV or Excel file |
| `target_col` | last column | Name of the target column |
| `test_size` | `0.2` | Share of data kept for testing |
| `use_nested_cv` | `true` | Use nested cross-validation |
| `remove_outliers` | `false` | Remove outliers with the IQR rule |
| `n_trials` | `30` | Number of Optuna tuning trials |
| `response_format` | `pdf` | `pdf` or `json` |

Example:

```bash
curl -X POST "http://127.0.0.1:8000/automl" \
  -F "file=@data.csv" \
  -F "target_col=price" \
  -F "response_format=json"
```

I also exposed it through ngrok and connected it to n8n so it can run inside automated workflows.

## Tech stack

FastAPI, Pandas, NumPy, scikit-learn, XGBoost, LightGBM, Optuna, category_encoders, Matplotlib/Seaborn

## Author

Hussain Ahmed · [LinkedIn](https://www.linkedin.com/in/hussian-ahmed) · hrfy330@gmail.com
