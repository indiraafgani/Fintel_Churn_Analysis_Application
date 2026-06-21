"""
FINTel Churn Model — Training Script
======================================
Mereplikasi PERSIS pipeline dari notebook fintel_churn_analysis_FINAL.ipynb:
  - Segmentasi Cold Start (tenure<=5) vs Non-Cold Start (tenure>5)
  - split_data() -> transformer (OHE + BinaryEncoder) -> RobustScaler -> SelectKBest
  - Benchmarking 12 model x 2 k x 2 resampler (FBeta2) -> top-3 -> RandomizedSearchCV tuning
  - Threshold optimization (F2-score) per segmen
  - SHAP (LinearExplainer/TreeExplainer auto-dispatch)
  - Export: model pipeline (.pkl), threshold, SHAP explainer artifacts, feature names

Catatan: hasil benchmarking notebook asli (LogisticRegression menang di kedua
segmen) sangat mungkin terulang di sini karena data identik, namun jika hasil
benchmarking berbeda (mis. karena versi library), script ini tetap memilih
model terbaik secara otomatis - bukan di-hardcode ke LogisticRegression.
"""

import time
import warnings
import pickle
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    VotingClassifier, RandomForestClassifier, StackingClassifier, BaggingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.preprocessing import RobustScaler, OneHotEncoder
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
)
from sklearn.metrics import (
    fbeta_score, make_scorer, recall_score, precision_score,
    precision_recall_curve, roc_auc_score, classification_report
)

import shap

warnings.filterwarnings('ignore')
RANDOM_STATE = 10
N_JOBS = 1  # sandbox has limited cores; set higher (-1) on a multi-core machine

# ──────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("FINTel Churn Model Training")
print("=" * 70)

df_full = pd.read_csv('df_full_with_id.csv')
print(f"Loaded df_full_with_id.csv: {df_full.shape}")

customer_ids = df_full['customerID'].copy()
df = df_full.drop(columns=['customerID'])  # mirrors notebook's `df` (raw merged, pre-customerID-drop)

df_clean = df.copy()

# Cold-start segmentation flag
df_clean['Is_Cold_Start'] = np.where(
    df_clean['tenure'] <= 5, 'Cold Start (<=5 bln)', 'Non-Cold Start (>5 bln)'
)

# add_on_count (EDA-engineered feature, used by risk-profile rules, NOT a model feature)
addon_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
              'TechSupport', 'StreamingTV', 'StreamingMovies']
df_clean['add_on_count'] = df_clean[addon_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)

# top10 city feature (engineered during EDA, dropped before modeling - kept here for parity)
top_10_cities = df_clean['City'].value_counts().head(10).index.tolist() if 'City' in df_clean.columns else []
if 'City' in df_clean.columns:
    df_clean['top10'] = df_clean['City'].apply(lambda x: x if x in top_10_cities else 'Others')

df_clean['Churn_binary'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

df_cold = df_clean[df_clean['Is_Cold_Start'] == 'Cold Start (<=5 bln)'].copy()
df_noncold = df_clean[df_clean['Is_Cold_Start'] == 'Non-Cold Start (>5 bln)'].copy()
print(f"Cold Start    : {df_cold.shape[0]} rows")
print(f"Non-Cold Start: {df_noncold.shape[0]} rows")

# ──────────────────────────────────────────────────────────────────────────
# 2. PIPELINE BUILDING BLOCKS (identik dengan notebook)
# ──────────────────────────────────────────────────────────────────────────

def split_data(dataFrame):
    drop_cols = ['Churn', 'ChurnValue', 'ChurnScore', 'ChurnReason', 'Churn_binary',
                 'Country', 'State', 'LatLong', 'Latitude', 'Longitude', 'ZipCode',
                 'City', 'top10', 'add_on_count', 'Is_Cold_Start']
    drop_cols = [c for c in drop_cols if c in dataFrame.columns]
    X = dataFrame.drop(columns=drop_cols)
    y = dataFrame['Churn'].map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


transformer = ColumnTransformer([
    ('cat_hot', OneHotEncoder(drop='first'), [
        'gender', 'SeniorCitizen', 'Partner',
        'Dependents', 'PhoneService', 'PaperlessBilling']),
    ('cat_bin', ce.BinaryEncoder(), [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'])
], remainder='passthrough')

ncr = NeighbourhoodCleaningRule()
smote = SMOTE(random_state=RANDOM_STATE)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg = LogisticRegression(random_state=RANDOM_STATE)
tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
knn = KNeighborsClassifier()
voting_soft = VotingClassifier([('clf1', logreg), ('clf2', tree), ('clf3', knn)], voting='soft')
voting_hard = VotingClassifier([('clf1', logreg), ('clf2', tree), ('clf3', knn)], voting='hard')
stack = StackingClassifier([('clf1', logreg), ('clf2', tree), ('clf3', knn)], final_estimator=tree)
bagging = BaggingClassifier(tree, random_state=RANDOM_STATE)
rf = RandomForestClassifier(random_state=RANDOM_STATE)
xgb = XGBClassifier(random_state=RANDOM_STATE)
lgbm = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
ridge = RidgeClassifier(random_state=RANDOM_STATE)

ALL_MODELS = [logreg, tree, knn, voting_hard, voting_soft, stack, bagging, rf, xgb, lgbm, ridge]

HYPERPARAM_REGISTRY = {
    'LogisticRegression': {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2', 'elasticnet'],
        'classifier__solver': ['saga'],
        'classifier__max_iter': [500, 1000],
        'classifier__class_weight': [None, 'balanced'],
    },
    'DecisionTreeClassifier': {
        'classifier__max_depth': [None, 5, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__class_weight': [None, 'balanced'],
    },
    'KNeighborsClassifier': {
        'classifier__n_neighbors': [3, 5, 7, 11, 15],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
        'classifier__leaf_size': [10, 20, 30],
    },
    'RandomForestClassifier': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': [None, 'balanced'],
    },
    'BaggingClassifier': {
        'classifier__n_estimators': [10, 50, 100],
        'classifier__max_samples': [0.5, 0.7, 1.0],
        'classifier__max_features': [0.5, 0.7, 1.0],
        'classifier__bootstrap': [True, False],
    },
    'XGBClassifier': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0],
        'classifier__scale_pos_weight': [1, 3, 5],
    },
    'LGBMClassifier': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [-1, 5, 10],
        'classifier__num_leaves': [31, 50, 100],
        'classifier__is_unbalance': [True, False],
        'classifier__class_weight': [None, 'balanced'],
    },
    'RidgeClassifier': {
        'classifier__alpha': [0.1, 1.0, 10.0, 100.0],
        'classifier__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag'],
        'classifier__class_weight': [None, 'balanced'],
    },
    'VotingClassifier': {
        'classifier__voting': ['soft', 'hard'],
    },
    'StackingClassifier': {
        'classifier__passthrough': [True, False],
    },
}


def top_3model(X_train, y_train):
    select_kbest = SelectKBest(score_func=f_classif, k=20)
    pipeline = ImbPipeline(steps=[
        ('preprocessor', transformer),
        ('scaler', RobustScaler()),
        ('feature_selection', select_kbest),
        ('resampler', smote),
        ('classifier', logreg)
    ])
    param_grid = [
        {'classifier': [model], 'feature_selection__k': [10, 20], 'resampler': [ncr, smote]}
        for model in ALL_MODELS
    ]
    fbeta_scorer = make_scorer(fbeta_score, beta=2)
    gscv = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, cv=skf, n_jobs=N_JOBS,
        scoring=fbeta_scorer, return_train_score=True, verbose=0
    )
    print("  Benchmarking (GridSearchCV)...")
    t0 = time.time()
    gscv.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s | Best: {gscv.best_score_:.4f}")

    cv_results_df = pd.DataFrame(gscv.cv_results_)
    cols = ['param_classifier', 'param_resampler', 'param_feature_selection__k',
            'mean_test_score', 'std_test_score', 'mean_fit_time']
    df_results = cv_results_df[cols].sort_values('mean_test_score', ascending=False)
    df_results['Model Name'] = df_results['param_classifier'].apply(lambda x: type(x).__name__)

    df_best_per_model = (
        df_results.groupby('Model Name', as_index=False)
        .apply(lambda g: g.loc[g['mean_test_score'].idxmax()])
        .reset_index(drop=True)
        .sort_values('mean_test_score', ascending=False)
    )
    top3_df = df_best_per_model.head(3)
    print(f"  Top-3: {top3_df['Model Name'].tolist()}")

    model_registry = {type(m).__name__: m for m in ALL_MODELS}
    top3_candidates = []
    for _, row in top3_df.iterrows():
        name = row['Model Name']
        top3_candidates.append({
            'name': name, 'model': model_registry[name],
            'resampler': row['param_resampler'], 'k': row['param_feature_selection__k'],
        })
    return top3_candidates


def tuning(top3_candidates, X_train, y_train):
    best_model_list = []
    for candidate in top3_candidates:
        name, model = candidate['name'], candidate['model']
        if name not in HYPERPARAM_REGISTRY:
            continue
        params = HYPERPARAM_REGISTRY[name]
        pipe = ImbPipeline([
            ('transform', transformer),
            ('scaler', RobustScaler()),
            ('resampler', candidate['resampler']),
            ('feature_select', SelectKBest(score_func=f_classif, k=candidate['k'])),
            ('classifier', model)
        ])
        rscv = RandomizedSearchCV(
            estimator=pipe, param_distributions=params, n_iter=100, cv=skf,
            scoring={'fbeta': make_scorer(fbeta_score, beta=2)},
            n_jobs=N_JOBS, random_state=RANDOM_STATE, refit='fbeta', verbose=0
        )
        print(f"  Tuning {name}...")
        rscv.fit(X_train, y_train)
        best_model_list.append({'name': name, 'model': rscv.best_estimator_, 'best_fbeta': rscv.best_score_})
        print(f"    -> FBeta(b=2)={rscv.best_score_:.4f}")

    best_entry = max(best_model_list, key=lambda x: x['best_fbeta'])
    print(f"  ## Model terbaik: {best_entry['name']} (FBeta={best_entry['best_fbeta']:.4f})")
    return best_entry['model']


def treshold(best_model, X_test, y_test):
    y_prob = best_model.predict_proba(X_test)[:, 1]
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)
    beta = 2
    f2_scores = ((1 + beta**2) * precision_vals[:-1] * recall_vals[:-1]) / \
                (beta**2 * precision_vals[:-1] + recall_vals[:-1] + 1e-10)
    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx]
    print(f"  Optimal threshold: {best_threshold:.4f} | F2={f2_scores[best_idx]:.4f} "
          f"| P={precision_vals[best_idx]:.4f} | R={recall_vals[best_idx]:.4f}")
    y_pred = (y_prob >= best_threshold).astype(int)
    return best_threshold, y_pred, y_prob


# ──────────────────────────────────────────────────────────────────────────
# 3. SHAP ENGINE (identik dengan notebook)
# ──────────────────────────────────────────────────────────────────────────

def get_feature_names(pipeline, X_original):
    try:
        col_transformer = pipeline.named_steps.get('preprocessor') or pipeline.named_steps.get('transform')
        raw_names = col_transformer.get_feature_names_out()
        all_names = [str(n).split('__', 1)[-1] for n in raw_names]
        selector = pipeline.named_steps.get('feature_select') or pipeline.named_steps.get('feature_selection')
        if selector and hasattr(selector, 'get_support'):
            mask = selector.get_support()
            if len(mask) == len(all_names):
                return [n for n, kept in zip(all_names, mask) if kept]
        return all_names
    except Exception as e:
        print(f"  [WARN] get_feature_names fallback: {e}")
        return list(X_original.columns)


def get_transformed_X(pipeline, X):
    X_transformed = X.copy()
    skip_steps = {'classifier', 'resampler'}
    for name, step in pipeline.steps:
        if name in skip_steps:
            continue
        X_transformed = step.transform(X_transformed)
    return X_transformed


TREE_MODELS = ('DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier',
               'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier')
LINEAR_MODELS = ('LogisticRegression', 'RidgeClassifier')


def build_shap_explainer(pipeline, X_test_original, sample_size=100):
    clf = None
    for name, step in pipeline.steps:
        if name == 'classifier':
            clf = step
    clf_name = type(clf).__name__
    X_trans = get_transformed_X(pipeline, X_test_original)
    feat_names = get_feature_names(pipeline, X_test_original)

    if not isinstance(X_trans, pd.DataFrame):
        X_trans = pd.DataFrame(X_trans, columns=feat_names)
    else:
        if list(X_trans.columns) != feat_names and len(feat_names) == X_trans.shape[1]:
            X_trans.columns = feat_names

    print(f"  SHAP model: {clf_name} | features: {X_trans.shape[1]}")

    if clf_name in TREE_MODELS:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        exp_type = 'TreeExplainer'
    elif clf_name in LINEAR_MODELS:
        masker = shap.maskers.Independent(X_trans)
        explainer = shap.LinearExplainer(clf, masker)
        shap_vals = explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        exp_type = 'LinearExplainer'
    else:
        predict_fn = (lambda x: clf.predict_proba(x)[:, 1]) if hasattr(clf, 'predict_proba') else clf.predict
        background = shap.sample(X_trans, min(sample_size, len(X_trans)), random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(predict_fn, background)
        eval_size = min(300, len(X_trans))
        shap_vals = explainer.shap_values(X_trans[:eval_size], nsamples=300)
        X_trans = X_trans.iloc[:eval_size]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        exp_type = 'KernelExplainer'

    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = expected_val[1]

    sv = np.array(shap_vals)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    return {
        'explainer': explainer, 'shap_values': sv, 'feature_names': feat_names,
        'explainer_type': exp_type, 'expected_value': float(expected_val),
        'model_name': clf_name,
    }


# ──────────────────────────────────────────────────────────────────────────
# 4. TRAIN BOTH SEGMENTS
# ──────────────────────────────────────────────────────────────────────────

def run_segment(df_seg, seg_name):
    print(f"\n{'='*70}\nSegment: {seg_name}\n{'='*70}")
    X_train, X_test, y_train, y_test = split_data(df_seg)
    top3 = top_3model(X_train, y_train)
    best_model = tuning(top3, X_train, y_train)
    best_threshold, y_pred, y_prob = treshold(best_model, X_test, y_test)

    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    model_name = type(best_model.named_steps['classifier']).__name__
    print(f"  ROC-AUC: {roc_auc:.4f} | model={model_name}")

    shap_result = build_shap_explainer(best_model, X_test)

    return {
        'model': best_model,
        'model_name': model_name,
        'threshold': float(best_threshold),
        'X_train_columns': list(X_train.columns),
        'metrics': {
            'recall': float(recall), 'precision': float(precision),
            'f2': float(f2), 'roc_auc': float(roc_auc),
            'n_train': len(X_train), 'n_test': len(X_test),
        },
        'shap': shap_result,
    }


results_cold = run_segment(df_cold, 'Cold Start (<=5 bln)')
results_noncold = run_segment(df_noncold, 'Non-Cold Start (>5 bln)')

# ──────────────────────────────────────────────────────────────────────────
# 5. SAVE ARTIFACTS
# ──────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}\nSaving artifacts\n{'='*70}")

with open('models/cold_start_model.pkl', 'wb') as f:
    pickle.dump({
        'model': results_cold['model'],
        'threshold': results_cold['threshold'],
        'model_name': results_cold['model_name'],
        'feature_columns': results_cold['X_train_columns'],
        'metrics': results_cold['metrics'],
    }, f)
print("Saved models/cold_start_model.pkl")

with open('models/noncold_model.pkl', 'wb') as f:
    pickle.dump({
        'model': results_noncold['model'],
        'threshold': results_noncold['threshold'],
        'model_name': results_noncold['model_name'],
        'feature_columns': results_noncold['X_train_columns'],
        'metrics': results_noncold['metrics'],
    }, f)
print("Saved models/noncold_model.pkl")

with open('models/cold_shap.pkl', 'wb') as f:
    pickle.dump({
        'explainer': results_cold['shap']['explainer'],
        'feature_names': results_cold['shap']['feature_names'],
        'explainer_type': results_cold['shap']['explainer_type'],
        'expected_value': results_cold['shap']['expected_value'],
        'model_name': results_cold['shap']['model_name'],
    }, f)
print("Saved models/cold_shap.pkl")

with open('models/noncold_shap.pkl', 'wb') as f:
    pickle.dump({
        'explainer': results_noncold['shap']['explainer'],
        'feature_names': results_noncold['shap']['feature_names'],
        'explainer_type': results_noncold['shap']['explainer_type'],
        'expected_value': results_noncold['shap']['expected_value'],
        'model_name': results_noncold['shap']['model_name'],
    }, f)
print("Saved models/noncold_shap.pkl")

# Save customer data (raw, with customerID + engineered cols needed by the app)
df_export = df_full.copy()
df_export['Is_Cold_Start'] = np.where(df_export['tenure'] <= 5, 'Cold Start', 'Non-Cold Start')
df_export['add_on_count'] = df_export[addon_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
df_export.to_csv('data/customer_data.csv', index=False)
print(f"Saved data/customer_data.csv ({df_export.shape})")

# Save model metadata summary (used by app sidebar "About Model")
metadata = {
    'cold_start': {
        'model_name': results_cold['model_name'],
        'threshold': results_cold['threshold'],
        'metrics': results_cold['metrics'],
        'shap_explainer': results_cold['shap']['explainer_type'],
    },
    'non_cold_start': {
        'model_name': results_noncold['model_name'],
        'threshold': results_noncold['threshold'],
        'metrics': results_noncold['metrics'],
        'shap_explainer': results_noncold['shap']['explainer_type'],
    },
    'median_monthly_charges': float(df_clean['MonthlyCharges'].median()),
    'median_add_on_count': float(df_clean['add_on_count'].median()),
    'gray_zone_lower': 54,
    'gray_zone_upper': 80,
    'cold_start_tenure_cutoff': 5,
}
with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("Saved models/model_metadata.json")
print(json.dumps(metadata, indent=2))

print("\nTraining complete.")
