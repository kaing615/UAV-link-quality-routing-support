from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.training.baselines.common import FEATURE_COLUMNS, evaluate_split, find_best_threshold
from src.training.baselines.RSSI_SNR_Baseline import ThresholdModel
from src.training.baselines.RSSI_SNR_Baseline import find_best_threshold as find_rssi_snr_thresholds

_MODEL_NAMES = {'xgb': 'XGBoost', 'mlp': 'Small MLP', 'logreg': 'Logistic Regression', 'rf': 'Random Forest', 'threshold': 'RSSI/SNR Threshold'}

def load_run_rows(run_name: str, splits: list[str] | None) -> pd.DataFrame:
    run_root = Path('data/graph_dataset') / run_name
    df = pd.read_csv(run_root / 'features' / 'edges_labeled.csv')
    if splits is not None:
        time_splits = pd.read_csv(run_root / 'splits' / 'time_splits.csv')
        keep_times = set(time_splits[time_splits['split'].isin(splits)]['time'])
        df = df[df['time'].isin(keep_times)]
    return df[FEATURE_COLUMNS + ['label']].reset_index(drop=True)

def build_model(model_id: str, pos_weight: float, random_state: int):
    if model_id == 'xgb':
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', eval_metric='logloss', scale_pos_weight=pos_weight, random_state=random_state)
    if model_id == 'mlp':
        return Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=500, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=random_state))])
    if model_id == 'logreg':
        return LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=random_state)
    if model_id == 'rf':
        return RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=random_state)
    raise ValueError(f'Unknown model_id: {model_id}')

def oversample_minority(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    counts = df['label'].value_counts()
    if len(counts) < 2:
        return df
    minority = counts.idxmin()
    n_extra = counts.max() - counts.min()
    extra = df[df['label'] == minority].sample(n=n_extra, replace=True, random_state=random_state)
    return pd.concat([df, extra], ignore_index=True)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Leave-one-run-out baseline training (xgb/mlp/logreg/rf/threshold).')
    p.add_argument('--test-run', type=str, required=True)
    p.add_argument('--train-runs', type=str, required=True, help='Comma-separated training run names')
    p.add_argument('--model', type=str, default='xgb', choices=list(_MODEL_NAMES.keys()))
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--random-state', type=int, default=42)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    model_id = args.model
    model_name = _MODEL_NAMES[model_id]
    train_runs = [r.strip() for r in args.train_runs.split(',') if r.strip()]
    output_dir = args.output_dir or Path('outputs/loro') / model_id / args.test_run
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'[LORO] model={model_id} | test_run={args.test_run}')
    print(f'       train_runs ({len(train_runs)}): {train_runs}')
    train_df = pd.concat([load_run_rows(r, ['train']) for r in train_runs], ignore_index=True)
    val_df = pd.concat([load_run_rows(r, ['val']) for r in train_runs], ignore_index=True)
    test_df = load_run_rows(args.test_run, None)
    print(f'       rows: train={len(train_df)} val={len(val_df)} test={len(test_df)}')
    n_pos = int((train_df['label'] == 1).sum())
    n_neg = int((train_df['label'] == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)
    if model_id == 'threshold':
        rssi_t, snr_t, _ = find_rssi_snr_thresholds(train_df)
        model = ThresholdModel(rssi_t, snr_t)
        threshold = 0.5
        print(f'[THR] rssi_thresh={rssi_t:.2f} snr_thresh={snr_t:.2f}')
    else:
        model = build_model(model_id, pos_weight, args.random_state)
        fit_df = oversample_minority(train_df, args.random_state) if model_id == 'mlp' else train_df
        model.fit(fit_df[FEATURE_COLUMNS], fit_df['label'])
        threshold, tuned_val_f1 = find_best_threshold(model, val_df)
        print(f'[THR] tuned threshold={threshold:.2f} (val macro_f1={tuned_val_f1:.4f})')
    val_metrics, val_preds = evaluate_split(model, model_id, model_name, val_df, 'val', threshold=threshold)
    test_metrics, test_preds = evaluate_split(model, model_id, model_name, test_df, 'test', threshold=threshold)
    pd.DataFrame([val_metrics, test_metrics]).to_csv(output_dir / 'metrics.csv', index=False)
    val_preds.to_csv(output_dir / 'val_predictions.csv', index=False)
    test_preds.to_csv(output_dir / 'test_predictions.csv', index=False)
    metadata = {'model_id': model_id, 'model_name': model_name, 'protocol': 'leave-one-run-out', 'test_run': args.test_run, 'train_runs': train_runs, 'feature_columns': FEATURE_COLUMNS, 'scale_pos_weight': pos_weight if model_id == 'xgb' else None, 'oversampled': model_id == 'mlp', 'class_weight': 'balanced' if model_id in ('logreg', 'rf') else None, 'threshold': threshold, 'rssi_thresh': float(model.rssi_thresh) if model_id == 'threshold' else None, 'snr_thresh': float(model.snr_thresh) if model_id == 'threshold' else None}
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f"[OK]  test ({args.test_run}): macro_f1={test_metrics['macro_f1']:.4f}  f1={test_metrics['f1']:.4f}  recall={test_metrics['recall']:.4f}")
    print(f'      outputs → {output_dir}')
if __name__ == '__main__':
    main()
