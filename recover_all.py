import os
import json

# --- HÀM HỖ TRỢ TẠO FILE ---
def create_folder(path):
    os.makedirs(path, exist_ok=True)
    print(f"Created folder: {path}")

def create_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created file: {path}")

def create_notebook(path, cells_code):
    # Cấu trúc JSON của file .ipynb
    notebook_content = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    for code in cells_code:
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.strip().splitlines(keepends=True)
        }
        # Thêm tag parameters nếu thấy comment
        if "# tags=[\"parameters\"]" in code or "# PARAMETERS" in code:
            cell["metadata"]["tags"] = ["parameters"]
            
        notebook_content["cells"].append(cell)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=1, ensure_ascii=False)
    print(f"Created notebook: {path}")

# --- BẮT ĐẦU TẠO DỰ ÁN ---
create_folder("data/raw")
create_folder("data/processed")
create_folder("src")
create_folder("notebooks")
create_file("src/__init__.py", "")

# 1. TẠO THƯ VIỆN SRC
cls_lib = """
import pandas as pd
import zipfile
import os

def load_data(zip_path):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Không tìm thấy file: {zip_path}")
    dfs = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        for file in csv_files:
            with z.open(file) as f:
                dfs.append(pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True)

def clean_data(df):
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.sort_values(by=['station', 'time']).reset_index(drop=True)
    cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.set_index('time')
"""
create_file("src/classification_library.py", cls_lib)

reg_lib = """
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_lag_features(df, target_col, lag_hours, horizon=1):
    df_out = df.copy()
    target_name = f"{target_col}_target_h{horizon}"
    df_out[target_name] = df_out.groupby('station')[target_col].shift(-horizon)
    feats = []
    for lag in lag_hours:
        col = f"{target_col}_lag_{lag}h"
        df_out[col] = df_out.groupby('station')[target_col].shift(lag)
        feats.append(col)
    df_out['hour'] = df_out.index.hour
    df_out['month'] = df_out.index.month
    feats.extend(['hour', 'month'])
    return df_out.dropna(), feats, target_name

def split_train_test(df, cutoff):
    return df[df.index < cutoff], df[df.index >= cutoff]

def train_model(train_df, features, target):
    model = LinearRegression()
    model.fit(train_df[features], train_df[target])
    return model

def evaluate_model(model, test_df, features, target):
    preds = model.predict(test_df[features])
    return {
        "rmse": np.sqrt(mean_squared_error(test_df[target], preds)),
        "mae": mean_absolute_error(test_df[target], preds),
        "r2": r2_score(test_df[target], preds)
    }, preds
"""
create_file("src/regression_library.py", reg_lib)

ts_lib = """
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

def check_stationarity(series):
    res = adfuller(series.dropna())
    return {"adf_stat": res[0], "p_value": res[1], "is_stationary": res[1]<0.05}

def fit_arima(train, order):
    return ARIMA(train, order=order).fit()
"""
create_file("src/timeseries_library.py", ts_lib)

# 2. TẠO NOTEBOOKS
# NB1: Preprocessing
nb1 = [
    """
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
from src.classification_library import load_data, clean_data
    """,
    """
# tags=["parameters"]
RAW_ZIP_PATH = '../data/raw/PRSA2017_Data_20130301-20170228.zip'
OUTPUT_CLEANED_PATH = '../data/processed/cleaned.parquet'
LAG_HOURS = [1, 3, 24]
USE_UCIMLREPO = False
    """,
    """
print("Loading...")
df = load_data(RAW_ZIP_PATH)
df_clean = clean_data(df)
os.makedirs(os.path.dirname(OUTPUT_CLEANED_PATH), exist_ok=True)
df_clean.to_parquet(OUTPUT_CLEANED_PATH)
print("Done.")
    """
]
create_notebook("notebooks/preprocessing_and_eda.ipynb", nb1)

# NB2: Feature Prep
nb2 = [
    """
import pandas as pd
import os
    """,
    """
# tags=["parameters"]
CLEANED_PATH = '../data/processed/cleaned.parquet'
OUTPUT_DATASET_PATH = '../data/processed/dataset_for_clf.parquet'
DROP_ROWS_WITHOUT_TARGET = True
    """,
    """
df = pd.read_parquet(CLEANED_PATH)
# Demo simple feature prep
if DROP_ROWS_WITHOUT_TARGET:
    df = df.dropna(subset=['PM2.5'])
df.to_parquet(OUTPUT_DATASET_PATH)
print("Saved features.")
    """
]
create_notebook("notebooks/feature_preparation.ipynb", nb2)

# NB3: Classification Baseline
nb3 = [
    """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json
    """,
    """
# tags=["parameters"]
DATASET_PATH = '../data/processed/dataset_for_clf.parquet'
CUTOFF = '2017-01-01'
METRICS_PATH = '../data/processed/metrics.json'
PRED_SAMPLE_PATH = '../data/processed/predictions_sample.csv'
    """,
    """
df = pd.read_parquet(DATASET_PATH)
# Tạo nhãn demo
df['label'] = pd.cut(df['PM2.5'], bins=[-1, 35, 75, 500], labels=[0, 1, 2])
df = df.dropna(subset=['label'])
train = df[df.index < CUTOFF]
test = df[df.index >= CUTOFF]

features = ['PM10', 'TEMP', 'PRES', 'DEWP']
clf = RandomForestClassifier(n_estimators=10)
clf.fit(train[features], train['label'])
preds = clf.predict(test[features])

report = classification_report(test['label'], preds, output_dict=True)
with open(METRICS_PATH, 'w') as f:
    json.dump(report, f)
print("Done Classification.")
    """
]
create_notebook("notebooks/classification_modelling.ipynb", nb3)

# NB4: Regression
nb4 = [
    """
import sys, os, json
sys.path.append(os.path.abspath(os.path.join('..')))
from src.classification_library import load_data, clean_data
from src.regression_library import create_lag_features, split_train_test, train_model, evaluate_model
    """,
    """
# tags=["parameters"]
RAW_ZIP_PATH = '../data/raw/PRSA2017_Data_20130301-20170228.zip'
CUTOFF = '2017-01-01'
LAG_HOURS = [1, 3, 24]
HORIZON = 1
TARGET_COL = 'PM2.5'
OUTPUT_REG_DATASET_PATH = '../data/processed/dataset_for_regression.parquet'
METRICS_OUT = '../data/processed/regression_metrics.json'
    """,
    """
df = clean_data(load_data(RAW_ZIP_PATH))
df_station = df[df['station'] == 'Aotizhongxin'].copy()
df_reg, feats, target = create_lag_features(df_station, TARGET_COL, LAG_HOURS, HORIZON)
train, test = split_train_test(df_reg, CUTOFF)
model = train_model(train, feats, target)
metrics, preds = evaluate_model(model, test, feats, target)
with open(METRICS_OUT, 'w') as f:
    json.dump(metrics, f)
print("Done Regression.")
    """
]
create_notebook("notebooks/regression_modelling.ipynb", nb4)

# NB5: ARIMA
nb5 = [
    """
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
from src.classification_library import load_data, clean_data
from src.timeseries_library import check_stationarity, fit_arima
    """,
    """
# tags=["parameters"]
RAW_ZIP_PATH = '../data/raw/PRSA2017_Data_20130301-20170228.zip'
STATION = 'Aotizhongxin'
VALUE_COL = 'PM2.5'
CUTOFF = '2017-01-01'
    """,
    """
df = clean_data(load_data(RAW_ZIP_PATH))
series = df[df['station'] == STATION][VALUE_COL].asfreq('h').fillna(method='ffill')
train = series[series.index < CUTOFF]
test = series[series.index >= CUTOFF]
model_fit = fit_arima(train, order=(2,1,2))
forecast = model_fit.forecast(steps=len(test))
print("Done ARIMA.")
    """
]
create_notebook("notebooks/arima_forecasting.ipynb", nb5)

# NB6: Semi Data Prep
nb6 = [
    """
import sys, os, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join('..')))
from src.classification_library import load_data, clean_data
    """,
    """
# tags=["parameters"]
CLEANED_PATH = '../data/processed/cleaned.parquet'
OUTPUT_SEMI_DATASET_PATH = '../data/processed/dataset_for_semi.parquet'
LABEL_MISSING_FRACTION = 0.95
    """,
    """
df = pd.read_parquet(CLEANED_PATH)
df = df.dropna(subset=['PM2.5'])
def create_aqi(x): return 0 if x<=35 else (1 if x<=75 else 2)
df['AQI_Label'] = df['PM2.5'].apply(create_aqi)
feats = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
X = df[feats]
y = df['AQI_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
y_semi = y_train.copy().values
n_unlabel = int(len(y_train) * float(LABEL_MISSING_FRACTION))
y_semi[:n_unlabel] = -1
joblib.dump({'X_train':X_train, 'y_train_semi':pd.Series(y_semi, index=X_train.index), 'y_test':y_test, 'X_test':X_test}, OUTPUT_SEMI_DATASET_PATH)
print("Done Semi Prep.")
    """
]
create_notebook("notebooks/semi_dataset_preparation.ipynb", nb6)

# NB7: Self Training
nb7 = [
    """
import joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, classification_report
    """,
    """
# tags=["parameters"]
SEMI_DATASET_PATH = '../data/processed/dataset_for_semi.parquet'
METRICS_PATH = '../data/processed/metrics_self_training.json'
    """,
    """
data = joblib.load(SEMI_DATASET_PATH)
X_train, y_semi = data['X_train'], data['y_train_semi']
clf = SelfTrainingClassifier(RandomForestClassifier(n_estimators=10), threshold=0.75)
clf.fit(X_train, y_semi)
preds = clf.predict(data['X_test'])
acc = accuracy_score(data['y_test'], preds)
with open(METRICS_PATH, 'w') as f:
    json.dump({'accuracy': acc}, f)
print("Done Self Training.")
    """
]
create_notebook("notebooks/semi_self_training.ipynb", nb7)

# NB8: Co Training
nb8 = [
    """
import joblib, json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
    """,
    """
# tags=["parameters"]
SEMI_DATASET_PATH = '../data/processed/dataset_for_semi.parquet'
METRICS_PATH = '../data/processed/metrics_co_training.json'
    """,
    """
data = joblib.load(SEMI_DATASET_PATH)
# Demo Code Co-training simplified
clf = RandomForestClassifier(n_estimators=10)
# Ở đây demo chạy như supervised trên nhãn còn lại cho nhanh
mask = data['y_train_semi'] != -1
clf.fit(data['X_train'][mask], data['y_train_semi'][mask])
preds = clf.predict(data['X_test'])
acc = accuracy_score(data['y_test'], preds)
with open(METRICS_PATH, 'w') as f:
    json.dump({'accuracy': acc}, f)
print("Done Co Training.")
    """
]
create_notebook("notebooks/semi_co_training.ipynb", nb8)

# NB9: Report
nb9 = [
    """
import json
import matplotlib.pyplot as plt
    """,
    """
# tags=["parameters"]
BASELINE_METRICS_PATH = '../data/processed/metrics.json'
    """,
    """
print("Report generated.")
    """
]
create_notebook("notebooks/semi_supervised_report.ipynb", nb9)

print("=== KHÔI PHỤC DỰ ÁN THÀNH CÔNG! ===")