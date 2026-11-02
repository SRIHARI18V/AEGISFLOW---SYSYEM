"""Champion (XGBoost) and alternative (LightGBM) fraud models.
Both take the same matrix and return raw probabilities; calibration is a separate
step so the two are comparable before and after it. scale_pos_weight, not
resampling: SMOTE on transaction data invents customers who never existed, and the
invented ones are exactly where the model then makes its mistakes.
"""
from __future__ import annotations

import numpy as np


def pos_weight(y) -> float:
    y = np.asarray(y, int)
    pos = max(int(y.sum()), 1)
    return float((len(y) - pos) / pos)


def train_xgboost(X_train, y_train, X_val, y_val, params: dict):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 800),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 5),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=pos_weight(y_train),
        eval_metric="aucpr", tree_method="hist",
        early_stopping_rounds=params.get("early_stopping_rounds", 60),
        random_state=params.get("seed", 42), n_jobs=params.get("n_jobs", 4))
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, model.predict_proba(X_val)[:, 1]


def train_lightgbm(X_train, y_train, X_val, y_val, params: dict):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=params.get("n_estimators", 800),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 63),
        min_child_samples=params.get("min_child_samples", 50),
        subsample=params.get("subsample", 0.8), subsample_freq=1,
        colsample_bytree=params.get("colsample_bytree", 0.8),
        scale_pos_weight=pos_weight(y_train),
        random_state=params.get("seed", 42), n_jobs=params.get("n_jobs", 4),
        verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(params.get("early_stopping_rounds", 60),
                                            verbose=False)])
    return model, model.predict_proba(X_val)[:, 1]


def shap_explainer(model, X_background):
    """Built at training time and saved with the model. M4's scoring service loads
    it once at startup so a per-request explanation costs about a millisecond
    instead of rebuilding the explainer on every call."""
    import shap
    return shap.TreeExplainer(model, data=None, feature_perturbation="tree_path_dependent")


def top_reason_codes(explainer, X_row, feature_names, k: int = 5) -> list[dict]:
    """The top-k features pushing this transaction's score UP, which is what a
    compliance officer has to be able to read (spec objective 5.3)."""
    values = np.asarray(explainer.shap_values(X_row)).reshape(-1)
    order = np.argsort(-values)[:k]
    return [{"feature": feature_names[i], "contribution": float(values[i])}
            for i in order if values[i] > 0]
