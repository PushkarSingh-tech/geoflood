"""Training + evaluation. Run: python -m ml.train [--csv path]"""
from __future__ import annotations
import argparse, json, time
import joblib, numpy as np, pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import DATA_DIR, CSV_NAME, MODEL_DIR, REPORT_DIR, TARGET, ID_COL, RISK_BINS
from . import preprocess as pp

SEED = 42


def _wrap(est, scale=False):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", est))
    return Pipeline(steps)


def _make_models(balance: bool):
    cw = "balanced" if balance else None
    models = {
        "dummy_prior": _wrap(DummyClassifier(strategy="prior")),
        "logistic_regression": _wrap(LogisticRegression(max_iter=1000, class_weight=cw), scale=True),
    }
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = _wrap(CatBoostClassifier(iterations=300, depth=4, verbose=0, random_seed=SEED,
                                                      auto_class_weights="Balanced" if balance else None))
        primary = "catboost"
    except Exception:
        models["hist_gradient_boosting"] = _wrap(HistGradientBoostingClassifier(
            max_depth=4, max_iter=200, learning_rate=0.05, random_state=SEED, class_weight=cw))
        primary = "hist_gradient_boosting"
    return models, primary


def split(df: pd.DataFrame):
    """Geographic split if a 'lat' column exists (northernmost 30% = test); else stratified random."""
    if "lat" in df.columns and df["lat"].notna().all():
        cut = df["lat"].quantile(0.70)
        test_mask = df["lat"] >= cut
        return df[~test_mask], df[test_mask], "geographic (train: southern 70% by latitude, test: northern 30%)"
    tr, te = train_test_split(df, test_size=0.3, stratify=df[TARGET], random_state=SEED)
    return tr, te, "random stratified 70/30 (no coordinates available; optimistic because of spatial autocorrelation)"


def _metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "pr_auc": float(average_precision_score(y, p)),
           "brier": float(brier_score_loss(y, p)),
           "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
           "threshold": 0.5}
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(set(y)) > 1 else None
    return out


def train(csv_path=None, model_dir=MODEL_DIR, report_dir=REPORT_DIR) -> dict:
    csv_path = csv_path or (DATA_DIR / CSV_NAME)
    raw = pp.load_csv(csv_path)
    report = pp.validate_dataframe(raw)
    if not report["ok"] or TARGET not in raw.columns:
        raise ValueError("Dataset not usable for training: " + "; ".join(report["errors"] or ["no target column"]))
    df = pp.engineer(pp.clean(raw)).dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    if df[TARGET].nunique() < 2:
        raise ValueError("Target has a single class; cannot train.")
    cols = pp.model_columns(df)
    tr, te, split_desc = split(df)
    if tr[TARGET].nunique() < 2 or te[TARGET].nunique() < 2:
        raise ValueError("Train or test split has one class only; need more data or a different split.")
    pos_rate = float(tr[TARGET].mean())
    balance = pos_rate < 0.20  # imbalance handling only when justified
    models, primary = _make_models(balance)

    results = {}
    for name, m in models.items():
        m.fit(tr[cols], tr[TARGET])
        results[name] = _metrics(te[TARGET].values, m.predict_proba(te[cols])[:, 1])

    best = models[primary]
    p = best.predict_proba(te[cols])[:, 1]
    frac_pos, mean_pred = calibration_curve(te[TARGET], p, n_bins=5, strategy="quantile")
    results[primary]["calibration"] = {"mean_predicted": mean_pred.round(3).tolist(),
                                       "fraction_positive": frac_pos.round(3).tolist()}
    imp = permutation_importance(best, te[cols], te[TARGET], scoring="average_precision",
                                 n_repeats=10, random_state=SEED)
    importance = sorted(({"feature": c, "importance": float(v)} for c, v in zip(cols, imp.importances_mean)),
                        key=lambda d: -d["importance"])

    # refit primary on all labelled rows for deployment; metrics above are from the held-out split only
    best.fit(df[cols], df[TARGET])

    src = raw["data_source"].astype(str) if "data_source" in raw.columns else pd.Series(["unknown"])
    synthetic = bool(src.str.upper().str.startswith("SYNTHETIC").any())
    meta = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "primary_model": primary,
            "features": cols, "target": TARGET, "target_meaning": "binary: waterlogging observed (1) or not (0)",
            "validation": split_desc, "n_train": int(len(tr)), "n_test": int(len(te)),
            "positive_rate_train": pos_rate, "class_imbalance_handling": "balanced class weights" if balance else "none (positive rate >= 20%)",
            "data_mode": "SYNTHETIC_DEMO" if synthetic else "USER_DATA",
            "risk_bins": [{"below": u, "category": c} for u, c in RISK_BINS],
            "notes": ["Held-out metrics come from the split above; the deployed model is refit on all labelled rows.",
                      "Score estimates the probability of the label's outcome, not flood depth or severity.",
                      "No separate validation set was used; no hyperparameter search was performed."]}
    model_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (report_dir / "metrics.json").write_text(json.dumps({"models": results, "validation": split_desc}, indent=2))
    (report_dir / "feature_importance.json").write_text(json.dumps(importance, indent=2))
    return {"meta": meta, "metrics": results, "importance": importance}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    out = train(a.csv)
    print(json.dumps({"validation": out["meta"]["validation"], "data_mode": out["meta"]["data_mode"]}, indent=2))
    for k, v in out["metrics"].items():
        print(f"{k:24s} PR-AUC={v['pr_auc']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")
