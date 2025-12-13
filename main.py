from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error, confusion_matrix,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder

# Models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

import optuna
import os
import tempfile
import warnings

warnings.filterwarnings('ignore')

plt.switch_backend('Agg')
sns.set_style('whitegrid')
app = FastAPI(title="Ultimate AutoML API", version="2.0")
def to_python(obj):
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# ==================== UTILITY FUNCTIONS ====================

def detect_task(y: pd.Series):
    """Smart task detection"""
    if y.dtype == "object":
        return "classification"
    if y.nunique() / len(y) < 0.05 or y.nunique() < 20:
        return "classification"
    return "regression"

def process_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from datetime columns"""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='raise')
            except:
                continue
        if np.issubdtype(df[col].dtype, np.datetime64):
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.weekday
            df[f"{col}_hour"] = df[col].dt.hour if hasattr(df[col].dt, 'hour') else 0
            df.drop(columns=[col], inplace=True)
    return df

def advanced_data_profiling(df: pd.DataFrame, target: str):
    """Comprehensive data analysis"""
    profile = {
        "shape": df.shape,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "target_info": {
            "name": target,
            "type": str(df[target].dtype),
            "unique_values": int(df[target].nunique()),
            "missing": int(df[target].isnull().sum())
        }
    }
    
    # Numeric statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        profile["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    return profile

def detect_outliers(df: pd.DataFrame, method='iqr'):
    """Detect outliers using IQR method"""
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outlier_count > 0:
            outliers[col] = {
                "count": int(outlier_count),
                "percentage": float(outlier_count / len(df) * 100)
            }
    
    return outliers

def intelligent_data_cleaning(df: pd.DataFrame, target: str, remove_outliers=False):
    """Smart data cleaning"""
    df = df.copy()
    
    # Remove ID columns
    id_cols = [c for c in df.columns if c != target and df[c].nunique() == len(df)]
    df = df.drop(columns=id_cols)
    
    # Remove high null columns (>80%)
    high_null_cols = [c for c in df.columns if c != target and df[c].isnull().sum() / len(df) > 0.8]
    df = df.drop(columns=high_null_cols)
    
    # Remove constant columns
    constant_cols = [c for c in df.columns if c != target and df[c].nunique() == 1]
    df = df.drop(columns=constant_cols)
    
    # Optionally remove outliers
    if remove_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    
    return df

def create_preprocessor(X, task="classification"):
    """Enhanced preprocessing pipeline"""
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    low_card_cat = [c for c in cat_features if X[c].nunique() <= 10]
    high_card_cat = [c for c in cat_features if X[c].nunique() > 10]
    
    transformers = []
    
    if num_features:
        transformers.append(
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler())  # Better for outliers
            ]), num_features)
        )
    
    if low_card_cat:
        transformers.append(
            ("cat_low", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20))
            ]), low_card_cat)
        )
    
    if high_card_cat:
        transformers.append(
            ("cat_high", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", TargetEncoder(smoothing=1.0))
            ]), high_card_cat)
        )
    
    return ColumnTransformer(transformers, remainder='drop')

def get_all_models(task="classification", random_state=42):
    """Get 10+ models for comparison"""
    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state, max_depth=10),
            "Random Forest": RandomForestClassifier(random_state=random_state, n_estimators=100, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(random_state=random_state, n_estimators=100, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state, n_estimators=100),
            "AdaBoost": AdaBoostClassifier(random_state=random_state, n_estimators=100),
            "XGBoost": XGBClassifier(random_state=random_state, n_jobs=-1, eval_metric="logloss"),
            "LightGBM": LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1),
            "K-Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(random_state=random_state, probability=True, max_iter=1000)
        }
        
        if CATBOOST_AVAILABLE:
            models["CatBoost"] = CatBoostClassifier(random_state=random_state, verbose=0)
    
    else:  # regression
        models = {
            "Linear Regression": Ridge(random_state=random_state),
            "Lasso": Lasso(random_state=random_state, max_iter=2000),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state, max_depth=10),
            "Random Forest": RandomForestRegressor(random_state=random_state, n_estimators=100, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(random_state=random_state, n_estimators=100, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state, n_estimators=100),
            "AdaBoost": AdaBoostRegressor(random_state=random_state, n_estimators=100),
            "XGBoost": XGBRegressor(random_state=random_state, n_jobs=-1),
            "LightGBM": LGBMRegressor(random_state=random_state, n_jobs=-1, verbosity=-1),
            "K-Neighbors": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
            "SVM": SVR(max_iter=1000)
        }
        
        if CATBOOST_AVAILABLE:
            models["CatBoost"] = CatBoostRegressor(random_state=random_state, verbose=0)
    
    return models

def plot_to_base64():
    """Convert plot to base64 string"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

# ==================== MAIN ENDPOINT ====================

@app.post("/automl")
async def automl(
    file: UploadFile = File(...),
    target_col: str = Form(None),
    test_size: float = Form(0.2),
    use_nested_cv: bool = Form(True),
    remove_outliers: bool = Form(False),
    n_trials: int = Form(30),
    response_format: str = Form("pdf")  # "pdf" or "json"
):
    """
    Ultimate AutoML with:
    - 10+ models comparison
    - Nested CV for best models
    - Optuna hyperparameter tuning
    - Comprehensive analysis
    
    response_format: "pdf" returns PDF report, "json" returns detailed JSON with base64 plots
    """
    
    # ==================== READ FILE ====================
    contents = await file.read()
    
    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(400, "Only CSV/Excel files supported")
    except Exception as e:
        raise HTTPException(400, f"Error reading file: {str(e)}")

    if df.shape[1] < 2:
        raise HTTPException(400, "Dataset must have at least 2 columns")
    if df.shape[0] < 20:
        raise HTTPException(400, "Dataset too small (need at least 20 rows)")
    
    # ==================== DATA PROFILING ====================
    target = target_col if target_col and target_col in df.columns else df.columns[-1]
    
    if df[target].isnull().sum() > 0:
        raise HTTPException(400, f"Target column '{target}' contains null values")
    
    data_profile = advanced_data_profiling(df, target)
    outliers_info = detect_outliers(df)
    
    # ==================== DATA CLEANING ====================
    y_raw = df[target].copy()
    X_raw = df.drop(columns=[target])
    
    X = process_dates(X_raw)
    X = intelligent_data_cleaning(pd.concat([X, y_raw], axis=1), target, remove_outliers)
    y_raw = X[target]
    X = X.drop(columns=[target])
    
    # ==================== TASK DETECTION ====================
    task = detect_task(y_raw)
    
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        class_mapping = {i: str(cls) for i, cls in enumerate(label_encoder.classes_)}
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        scoring = "accuracy"
    else:
        y = y_raw.astype(float).values
        label_encoder = None
        class_mapping = None
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=123)
        scoring = "r2"
    
    # ==================== TRAIN/TEST SPLIT ====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if task == "classification" else None
    )
    
    preprocessor = create_preprocessor(X_train, task)
    
    # ==================== MODEL COMPARISON (ALL MODELS) ====================
    print("🔍 Comparing 10+ models...")
    all_models = get_all_models(task)
    
    model_scores = {}
    for name, model in all_models.items():
        try:
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            scores = cross_val_score(pipe, X_train, y_train, cv=inner_cv, scoring=scoring, n_jobs=-1)
            model_scores[name] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "scores": scores.tolist()
            }
            print(f"✓ {name}: {scores.mean():.4f} (±{scores.std():.4f})")
        except Exception as e:
            print(f"✗ {name}: Failed - {str(e)}")
            model_scores[name] = {"mean": 0.0, "std": 0.0, "scores": []}
    
    # Select top 3 models for nested CV
    top_3_models = sorted(model_scores.items(), key=lambda x: x[1]["mean"], reverse=True)[:3]
    top_model_names = [name for name, _ in top_3_models]
    
    print(f"\n🏆 Top 3 models: {top_model_names}")
    
    # ==================== NESTED CV (TOP 3 MODELS) ====================
    nested_scores = {}
    
    if use_nested_cv and len(X_train) >= 100:
        print("\n🔄 Running Nested CV on top 3 models...")
        
        for model_name in top_model_names:
            outer_scores = []
            base_model = all_models[model_name]
            
            for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train if task == "classification" else None)):
                X_outer_train = X_train.iloc[train_idx]
                X_outer_val = X_train.iloc[val_idx]
                y_outer_train = y_train[train_idx]
                y_outer_val = y_train[val_idx]
                
                # Simple training for non-tunable models
                if model_name in ["Naive Bayes", "K-Neighbors", "SVM", "Logistic Regression", "Linear Regression", "Lasso"]:
                    pipe = Pipeline([("prep", preprocessor), ("model", base_model)])
                    pipe.fit(X_outer_train, y_outer_train)
                    
                    if task == "classification":
                        score = accuracy_score(y_outer_val, pipe.predict(X_outer_val))
                    else:
                        score = r2_score(y_outer_val, pipe.predict(X_outer_val))
                else:
                    # Hyperparameter tuning for tree-based models
                    def objective(trial):
                        if "XGBoost" in model_name:
                            params = {
                                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                                "max_depth": trial.suggest_int("max_depth", 3, 8),
                                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                            }
                            model = XGBClassifier(**params, random_state=42, n_jobs=-1, eval_metric="logloss") if task == "classification" else XGBRegressor(**params, random_state=42, n_jobs=-1)
                        
                        elif "LightGBM" in model_name:
                            params = {
                                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                            }
                            model = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbosity=-1) if task == "classification" else LGBMRegressor(**params, random_state=42, n_jobs=-1, verbosity=-1)
                        
                        elif "CatBoost" in model_name and CATBOOST_AVAILABLE:
                            params = {
                                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                                "depth": trial.suggest_int("depth", 3, 8),
                                "iterations": trial.suggest_int("iterations", 50, 300),
                            }
                            model = CatBoostClassifier(**params, random_state=42, verbose=0) if task == "classification" else CatBoostRegressor(**params, random_state=42, verbose=0)
                        
                        else:  # Random Forest, Extra Trees, etc.
                            params = {
                                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                                "max_depth": trial.suggest_int("max_depth", 3, 20),
                                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                            }
                            if "Random Forest" in model_name:
                                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1) if task == "classification" else RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                            elif "Extra Trees" in model_name:
                                model = ExtraTreesClassifier(**params, random_state=42, n_jobs=-1) if task == "classification" else ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)
                            else:
                                model = base_model
                        
                        pipe = Pipeline([("prep", preprocessor), ("model", model)])
                        return cross_val_score(pipe, X_outer_train, y_outer_train, cv=3, scoring=scoring).mean()
                    
                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=10, show_progress_bar=False)
                    
                    # Get best model
                    best_params = study.best_params
                    if "XGBoost" in model_name:
                        final_model = XGBClassifier(**best_params, random_state=42, n_jobs=-1, eval_metric="logloss") if task == "classification" else XGBRegressor(**best_params, random_state=42, n_jobs=-1)
                    elif "LightGBM" in model_name:
                        final_model = LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbosity=-1) if task == "classification" else LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=-1)
                    elif "CatBoost" in model_name and CATBOOST_AVAILABLE:
                        final_model = CatBoostClassifier(**best_params, random_state=42, verbose=0) if task == "classification" else CatBoostRegressor(**best_params, random_state=42, verbose=0)
                    elif "Random Forest" in model_name:
                        final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1) if task == "classification" else RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
                    elif "Extra Trees" in model_name:
                        final_model = ExtraTreesClassifier(**best_params, random_state=42, n_jobs=-1) if task == "classification" else ExtraTreesRegressor(**best_params, random_state=42, n_jobs=-1)
                    else:
                        final_model = base_model
                    
                    pipe = Pipeline([("prep", preprocessor), ("model", final_model)])
                    pipe.fit(X_outer_train, y_outer_train)
                    
                    if task == "classification":
                        score = accuracy_score(y_outer_val, pipe.predict(X_outer_val))
                    else:
                        score = r2_score(y_outer_val, pipe.predict(X_outer_val))
                
                outer_scores.append(score)
            
            nested_scores[model_name] = {
                "mean": float(np.mean(outer_scores)),
                "std": float(np.std(outer_scores)),
                "scores": [float(s) for s in outer_scores]
            }
            print(f"✓ {model_name} Nested CV: {np.mean(outer_scores):.4f} (±{np.std(outer_scores):.4f})")
        
        best_name = max(nested_scores, key=lambda k: nested_scores[k]["mean"])
    else:
        best_name = top_model_names[0]
        nested_scores = {name: model_scores[name] for name in top_model_names}
    
    print(f"\n🎯 Selected Model: {best_name}")
    
    # ==================== FINAL MODEL TRAINING WITH OPTUNA ====================
    print(f"\n⚙️ Fine-tuning {best_name} with Optuna...")
    
    best_base_model = all_models[best_name]
    
    if best_name in ["Naive Bayes", "K-Neighbors", "SVM", "Logistic Regression", "Linear Regression", "Lasso"]:
        final_pipeline = Pipeline([("prep", preprocessor), ("model", best_base_model)])
        final_pipeline.fit(X_train, y_train)
        final_params = {}
    else:
        def final_objective(trial):
            if "XGBoost" in best_name:
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
                model = XGBClassifier(**params, random_state=42, n_jobs=-1, eval_metric="logloss") if task == "classification" else XGBRegressor(**params, random_state=42, n_jobs=-1)
            
            elif "LightGBM" in best_name:
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 50),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                }
                model = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbosity=-1) if task == "classification" else LGBMRegressor(**params, random_state=42, n_jobs=-1, verbosity=-1)
            
            elif "CatBoost" in best_name and CATBOOST_AVAILABLE:
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "depth": trial.suggest_int("depth", 3, 10),
                    "iterations": trial.suggest_int("iterations", 100, 1000),
                }
                model = CatBoostClassifier(**params, random_state=42, verbose=0) if task == "classification" else CatBoostRegressor(**params, random_state=42, verbose=0)
            
            else:  # Random Forest, Extra Trees, etc.
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                }
                if "Random Forest" in best_name:
                    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1) if task == "classification" else RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                elif "Extra Trees" in best_name:
                    model = ExtraTreesClassifier(**params, random_state=42, n_jobs=-1) if task == "classification" else ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)
                else:
                    model = best_base_model
            
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            return cross_val_score(pipe, X_train, y_train, cv=inner_cv, scoring=scoring).mean()
        
        final_study = optuna.create_study(direction="maximize")
        final_study.optimize(final_objective, n_trials=n_trials, show_progress_bar=False)
        
        final_params = final_study.best_params
        print(f"Best params: {final_params}")
        
        # Create final model
        if "XGBoost" in best_name:
            final_model = XGBClassifier(**final_params, random_state=42, n_jobs=-1, eval_metric="logloss") if task == "classification" else XGBRegressor(**final_params, random_state=42, n_jobs=-1)
        elif "LightGBM" in best_name:
            final_model = LGBMClassifier(**final_params, random_state=42, n_jobs=-1, verbosity=-1) if task == "classification" else LGBMRegressor(**final_params, random_state=42, n_jobs=-1, verbosity=-1)
        elif "CatBoost" in best_name and CATBOOST_AVAILABLE:
            final_model = CatBoostClassifier(**final_params, random_state=42, verbose=0) if task == "classification" else CatBoostRegressor(**final_params, random_state=42, verbose=0)
        elif "Random Forest" in best_name:
            final_model = RandomForestClassifier(**final_params, random_state=42, n_jobs=-1) if task == "classification" else RandomForestRegressor(**final_params, random_state=42, n_jobs=-1)
        elif "Extra Trees" in best_name:
            final_model = ExtraTreesClassifier(**final_params, random_state=42, n_jobs=-1) if task == "classification" else ExtraTreesRegressor(**final_params, random_state=42, n_jobs=-1)
        else:
            final_model = best_base_model
        
        final_pipeline = Pipeline([("prep", preprocessor), ("model", final_model)])
        final_pipeline.fit(X_train, y_train)
    
    # ==================== PREDICTIONS & METRICS ====================
    train_preds = final_pipeline.predict(X_train)
    test_preds = final_pipeline.predict(X_test)
    
    if task == "classification":
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, test_preds, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)
        
        try:
            test_auc = roc_auc_score(y_test, final_pipeline.predict_proba(X_test), multi_class='ovr', average='weighted')
        except:
            test_auc = 0.0
        
        metrics = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "test_auc": float(test_auc)
        }
        overfitting_gap = train_acc - test_acc
    else:
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mape = np.mean(np.abs((y_test - test_preds) / (y_test + 1e-10))) * 100
        
        metrics = {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "test_mape": float(test_mape)
        }
        overfitting_gap = train_r2 - test_r2
    
    # ==================== VISUALIZATIONS ====================
    plots = {}
    
    # 1. Model Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = list(model_scores.keys())
    model_means = [model_scores[m]["mean"] for m in model_names]
    colors = ['green' if m == best_name else 'skyblue' for m in model_names]
    bars = ax.barh(model_names, model_means, color=colors)
    ax.set_xlabel(scoring.upper())
    ax.set_title("Model Comparison (Cross-Validation Scores)", fontsize=14, weight='bold')
    ax.grid(axis='x', alpha=0.3)
    plots["model_comparison"] = plot_to_base64()
    
    # 2. Target Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    if task == "classification":
        sns.countplot(x=y_raw, ax=ax, palette='viridis')
        ax.set_title("Target Distribution", fontsize=14, weight='bold')
    else:
        sns.histplot(y_raw, kde=True, ax=ax, bins=30, color='steelblue')
        ax.set_title("Target Distribution", fontsize=14, weight='bold')
    plots["target_distribution"] = plot_to_base64()
    
    # 3. Feature Importance
    try:
        importances = final_pipeline.named_steps["model"].feature_importances_
        fnames = final_pipeline.named_steps["prep"].get_feature_names_out()
        imp_df = pd.DataFrame({
            "feature": fnames,
            "importance": importances
        }).sort_values("importance", ascending=False).head(20)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, palette="rocket")
        ax.set_title("Top 20 Feature Importances", fontsize=14, weight='bold')
        plots["feature_importance"] = plot_to_base64()
    except:
        plots["feature_importance"] = None
    
    # 4. Performance Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if task == "regression":
        axes[0].scatter(y_test, test_preds, alpha=0.6, edgecolors='k', s=50)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel("Actual", fontsize=12)
        axes[0].set_ylabel("Predicted", fontsize=12)
        axes[0].set_title("Test: Actual vs Predicted", fontsize=14, weight='bold')
        axes[0].grid(alpha=0.3)
        
        residuals = y_test - test_preds
        axes[1].scatter(test_preds, residuals, alpha=0.6, edgecolors='k', s=50)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel("Predicted", fontsize=12)
        axes[1].set_ylabel("Residuals", fontsize=12)
        axes[1].set_title("Residual Plot", fontsize=14, weight='bold')
        axes[1].grid(alpha=0.3)
    else:
        cm_test = confusion_matrix(y_test, test_preds)
        sns.heatmap(cm_test, annot=True, fmt='d', ax=axes[0], cmap='Blues', cbar=True)
        axes[0].set_title("Test Confusion Matrix", fontsize=14, weight='bold')
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        
        cm_train = confusion_matrix(y_train, train_preds)
        sns.heatmap(cm_train, annot=True, fmt='d', ax=axes[1], cmap='Greens', cbar=True)
        axes[1].set_title("Train Confusion Matrix", fontsize=14, weight='bold')
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
    
    plt.tight_layout()
    plots["performance"] = plot_to_base64()
    
    # 5. Correlation Heatmap (if numeric features exist)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Heatmap", fontsize=14, weight='bold')
        plots["correlation"] = plot_to_base64()
    else:
        plots["correlation"] = None
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 6. Missing Values
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_data[missing_data > 0].sort_values(ascending=False).plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel("Missing Count")
        ax.set_title("Missing Values by Feature", fontsize=14, weight='bold')
        plots["missing_values"] = plot_to_base64()
    else:
        plots["missing_values"] = None
        json_path = os.path.join(OUTPUT_DIR, "AutoML_Result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            import json
            json.dump(
                to_python({
                    "status": "success",
                    "task": task,
                    "target": target,
                    "best_model": best_name,
                    "metrics": metrics,
                    "overfitting_gap": float(overfitting_gap)
                    }),
                f,
                ensure_ascii=False,
                indent=4
                )
    # ==================== RESPONSE FORMAT ====================
    if response_format == "json":
        return JSONResponse(
    to_python({
        "status": "success",
        "data_profile": data_profile,
        "outliers": outliers_info,
        "task": task,
        "target": target,
        "best_model": best_name,
        "best_params": final_params if final_params else {},
        "all_models_scores": model_scores,
        "nested_cv_scores": nested_scores if nested_scores else {},
        "metrics": metrics,
        "overfitting_gap": float(overfitting_gap),
        "overfitting_status": "High" if overfitting_gap > 0.15 else "Moderate" if overfitting_gap > 0.08 else "Good",
        "plots": {k: v for k, v in plots.items() if v is not None},
        "feature_count": X.shape[1],
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }))
    
    else:  # PDF Report
        OUTPUT_DIR = "outputs"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pdf_path = os.path.join(OUTPUT_DIR, "AutoML_Report.pdf")
        try:
            with PdfPages(pdf_path) as pdf:
                # Page 1: Executive Summary
                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                y_pos = 0.95
                ax.text(0.5, y_pos, "🤖 Ultimate AutoML Report", fontsize=24, ha='center', weight='bold')
                y_pos -= 0.08
                
                ax.text(0.5, y_pos, f"Task: {task.upper()} | Target: {target}", fontsize=14, ha='center')
                y_pos -= 0.06
                
                cv_method = f"Nested CV (5×3 folds)" if use_nested_cv else "Simple CV"
                ax.text(0.5, y_pos, f"CV Method: {cv_method} | Models Tested: {len(model_scores)}", 
                       fontsize=11, ha='center', style='italic')
                y_pos -= 0.08
                
                # Model Comparison
                ax.text(0.5, y_pos, "📊 Model Comparison (All Models)", fontsize=13, ha='center', weight='bold')
                y_pos -= 0.04
                
                for name, score_info in sorted(model_scores.items(), key=lambda x: x[1]["mean"], reverse=True)[:10]:
                    label = "🏆" if name == best_name else "  "
                    ax.text(0.5, y_pos, f"{label} {name}: {score_info['mean']:.4f} (±{score_info['std']:.4f})",
                           fontsize=10, ha='center', weight='bold' if name == best_name else 'normal')
                    y_pos -= 0.035
                
                y_pos -= 0.03
                ax.text(0.5, y_pos, f"✅ Selected Model: {best_name}", 
                       fontsize=13, ha='center', weight='bold', color='green')
                y_pos -= 0.06
                
                # Metrics
                ax.text(0.5, y_pos, "📈 Final Performance Metrics", fontsize=13, ha='center', weight='bold')
                y_pos -= 0.04
                
                for metric_name, metric_val in metrics.items():
                    ax.text(0.5, y_pos, f"{metric_name}: {metric_val:.4f}", fontsize=10, ha='center')
                    y_pos -= 0.035
                
                # Overfitting Status
                y_pos -= 0.03
                if overfitting_gap > 0.15:
                    status, color = "⚠️ High Overfitting Detected", "red"
                elif overfitting_gap > 0.08:
                    status, color = "⚠️ Moderate Overfitting", "orange"
                else:
                    status, color = "✅ Good Generalization", "green"
                
                ax.text(0.5, y_pos, f"Overfitting Status: {status}", 
                       fontsize=11, ha='center', weight='bold', color=color)
                ax.text(0.5, y_pos - 0.035, f"Train-Test Gap: {overfitting_gap:.4f}", 
                       fontsize=9, ha='center', style='italic')
                y_pos -= 0.08
                
                # Dataset Info
                ax.text(0.5, y_pos, f"Dataset: {len(df)} rows × {df.shape[1]} columns", 
                       fontsize=9, ha='center', style='italic')
                ax.text(0.5, y_pos - 0.03, f"Train: {len(X_train)} | Test: {len(X_test)}", 
                       fontsize=9, ha='center', style='italic')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # Add visualization pages
                for plot_name, plot_base64 in plots.items():
                    if plot_base64:
                        fig = plt.figure(figsize=(11, 8.5))
                        ax = fig.add_subplot(111)
                        img_data = base64.b64decode(plot_base64)
                        img = plt.imread(io.BytesIO(img_data))
                        ax.imshow(img)
                        ax.axis('off')
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
            
            return FileResponse(
                pdf_path,
                media_type="application/pdf",
                filename="Ultimate_AutoML_Report.pdf"
            )
        
        except Exception as e:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            raise HTTPException(500, f"Error generating report: {str(e)}")

@app.get("/")
async def root():
    return {
        "title": "🤖 Ultimate AutoML API",
        "version": "2.0",
        "features": [
            "10+ Machine Learning Models",
            "Nested Cross-Validation",
            "Optuna Hyperparameter Tuning",
            "Comprehensive Data Profiling",
            "Outlier Detection",
            "Advanced Preprocessing",
            "PDF Reports & JSON API"
        ],
        "endpoint": "/automl",
        "parameters": {
            "file": "CSV or Excel file (required)",
            "target_col": "Target column name (optional, default: last column)",
            "test_size": "Test set ratio (optional, default: 0.2)",
            "use_nested_cv": "Use nested CV for top models (optional, default: true)",
            "remove_outliers": "Remove outliers automatically (optional, default: false)",
            "n_trials": "Optuna trials for hyperparameter tuning (optional, default: 30)",
            "response_format": "'pdf' or 'json' (optional, default: 'pdf')"
        }
    }
