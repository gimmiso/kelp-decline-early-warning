"""Modeling utilities for explainable kelp decline early-warning models."""

import pandas as pd
import shap
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def split_features_target(df, feature_cols, target_col, test_size=0.2, random_state=42):
    """Split a modeling table into train and test sets."""
    x = df[feature_cols]
    y = df[target_col]
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


def train_xgboost_classifier(x_train, y_train, random_state=42, **kwargs):
    """Train a baseline XGBoost classifier."""
    params = {
        "random_state": random_state,
        "eval_metric": "logloss",
    }
    params.update(kwargs)
    model = XGBClassifier(**params)
    model.fit(x_train, y_train)
    return model


def evaluate_classifier(model, x_test, y_test):
    """Return common classification metrics for a fitted model."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }


def compute_shap_values(model, features):
    """Compute SHAP values for a fitted tree-based model."""
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(features)


def feature_importance_table(model, feature_names):
    """Return feature importances as a sorted table."""
    return (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
