import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os

# 1. 讀取資料

def load_instance(path):
    """讀取 AI4I 2020 資料集"""
    return pd.read_csv(path)

# 2. 前處理
def preprocess(df):
    """資料前處理，產生特徵與標籤"""
    df_processed = df.drop(columns=['UDI', 'Product ID', 'TWF', 'PWF', 'RNF', 'HDF', 'OSF'], errors='ignore')
    priority = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    if set(priority).issubset(df.columns):
        def map_failure(row):
            for mode in priority:
                if row[mode] == 1:
                    return mode
            return 'No Failure'
        df_processed['Failure_type'] = df.apply(map_failure, axis=1)
        df_processed['Machine_Failure'] = df_processed['Failure_type'].apply(lambda x: 0 if x == 'No Failure' else 1)
        df_processed = df_processed.drop(columns=priority + ['Machine failure'], errors='ignore')
    else:
        df_processed['Machine_Failure'] = df['Machine failure']
        df_processed['Failure_type'] = df['Machine failure'].apply(lambda x: 'No Failure' if x == 0 else 'Failure')
    df_processed['Failure_type_labels'] = df_processed['Failure_type'].astype('category').cat.codes

    # ——計算 Tool wear ——
    df_processed['Tool wear'] = df_processed['Tool wear'].replace(0, 1e-6)

    # ——計算 Power、PowerWear、TempPerPower ——
    df_processed['Power'] = df_processed['Rotational speed'] * df_processed['Torque']
    df_processed['Power'] = df_processed['Power'].replace(0, 1e-6)
    df_processed['PowerWear'] = df_processed['Power'] * df_processed['Tool wear']
    df_processed['TempPerPower'] = df_processed['Process temperature'] / df_processed['Power']

    # ——對 Power、PowerWear 做 log1p 變換（壓縮量級）——
    # df_processed['Power'] = np.log1p(df_processed['Power'])
    # df_processed['PowerWear'] = np.log1p(df_processed['PowerWear'])

    df_processed = df_processed[['Air temperature', 'Rotational speed', 'Tool wear',
                                 'Torque', 'Process temperature', 'Power', 'PowerWear', 'TempPerPower',
                                 'Machine_Failure', 'Failure_type', 'Failure_type_labels']]
    return df_processed

# 3. 分割資料

def split_train_test(df_processed):
    """分割訓練/測試集 (二元與多類別)"""
    X_bin = df_processed[['Air temperature', 'Rotational speed', 'Tool wear',
                          'Torque', 'Process temperature', 'Power', 'PowerWear', 'TempPerPower']]
    y_bin = df_processed['Machine_Failure']
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin, y_bin, stratify=y_bin, test_size=0.2, random_state=42)
    X_multi = X_bin.copy()
    y_multi = df_processed['Failure_type_labels']
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, stratify=y_multi, test_size=0.2, random_state=42)
    return (X_train_bin, X_test_bin, y_train_bin, y_test_bin,
            X_train_multi, X_test_multi, y_train_multi, y_test_multi)

# 4. 定義模型與參數

def get_models_and_params():
    """
    取得模型與參數網格 (二元/多類別)
    KNN、SVM、MLP 會自動標準化。
    """
    # 1. 構建模型字典：僅對 KNN、SVM、MLP 加入標準化步驟
    models = {
        'KNN': Pipeline([
            ('scaler', RobustScaler()),
            ('clf', KNeighborsClassifier())
        ]),
        'SVM': Pipeline([
            ('scaler', RobustScaler()),
            ('clf', SVC(probability=False, random_state=42))
        ]),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        'MLP': Pipeline([
            ('scaler', RobustScaler()),
            ('clf', MLPClassifier(max_iter=300, random_state=42))
        ])
    }

    # 2. 每個模型對應的參數搜尋空間
    param_grids = {
        'KNN': {
            'clf__n_neighbors': [3, 5, 7],
            'clf__weights': ['uniform', 'distance']
        },
        'SVM': {
            'clf__C': [0.1, 1, 10],
            'clf__kernel': ['linear', 'rbf']
        },
        'DecisionTree': {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        },
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 5]
        },
        'XGBoost': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.01]
        },
        'MLP': {
            'clf__hidden_layer_sizes': [(50,), (100,)],
            'clf__alpha': [1e-4, 1e-3]
        }
    }
    return models, param_grids

# 5. 訓練與評估

def evaluate(X_train, X_test, y_train, y_test, task='binary'):
    """模型訓練、超參數搜尋與評估 (含進度條)"""
     # 隱藏所有 warning
    warnings.filterwarnings('ignore')
    np.seterr(all='ignore')
    models, param_grids = get_models_and_params()
    results = {}
    for name, base_model in tqdm(models.items(), desc="Tuning Models"):
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grids[name],
            scoring='f1_macro',
            cv=5,
            n_jobs=-1,
            verbose=0
        )
        # print(f"\n===== Tuning {name} =====")
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        # print(f">>> Best params for {name}: {grid.best_params_}\n")
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        results[name] = {
            'Best_Params': grid.best_params_,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        }
    return pd.DataFrame(results).T

# 6. 視覺化

def plot_f1_comparison(results_bin, results_multi, fname=None):
    """繪製二元與多類別 F1 分數比較圖，標註 fname 並存到 /plot"""
    df_plot = pd.DataFrame({
        'Binary': results_bin['F1'],
        'Multi-class': results_multi['F1']
    })
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = list(range(len(df_plot)))
    bars_binary = plt.bar(
        [i - bar_width/2 for i in index],
        df_plot['Binary'],
        width=bar_width,
        label='Binary',
        color='darkgray'
    )
    bars_multi = plt.bar(
        [i + bar_width/2 for i in index],
        df_plot['Multi-class'],
        width=bar_width,
        label='Multi-class',
        color='black'
    )
    for bar in bars_binary:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    for bar in bars_multi:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Macro F1 Score', fontsize=12)
    plt.title(f'Comparison of Macro F1 Score: Binary vs Multi-class on {fname}', fontsize=14)
    plt.xticks(index, df_plot.index, fontsize=11)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if fname is not None:
        os.makedirs('plot', exist_ok=True)
        plt.savefig(f"plot/{fname}_f1_comparison.png", dpi=200)
    plt.show()
