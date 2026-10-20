"""CLI training entrypoint for OpsPilot AI SLA Risk Model."""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.loader import DataLoader
from src.pipeline.features import FeatureEngineer
from src.models.sla_model import SLARiskModel
from src.config import MODELS_DIR


def main():
    print("=" * 70)
    print("[OpsPilot AI] Training SLA Risk Model (Pre-Assignment)")
    print("=" * 70)

    # 1. Ingestion
    print("\n[1/5] Loading and validating operational datasets...")
    loader = DataLoader()
    tickets = loader.load_tickets()
    merchants = loader.load_merchants()
    hist_tickets, open_tickets = loader.split_tickets(tickets)
    print(f"  - Historical completed tickets: {len(hist_tickets):,}")
    print(f"  - Live open queue tickets: {len(open_tickets):,} (excluded from training)")

    # 2. Feature Engineering (Strictly Pre-Assignment)
    print("\n[2/5] Preparing leakage-free intake feature matrices...")
    fe = FeatureEngineer()
    X, y = fe.prepare_training_dataset(hist_tickets, merchants)
    print(f"  - Training feature matrix X: {X.shape}")
    print(f"  - Target distribution: {y.mean():.1%} SLA breach rate ({y.sum()} breaches / {len(y)} tickets)")

    # 3. Model Training, CV Benchmarking & Calibration
    print("\n[3/5] Running 5-Fold Stratified CV & Holdout Evaluation...")
    model = SLARiskModel(model_type="random_forest", calibrate=True)
    metadata = model.fit_and_evaluate(X, y, test_size=0.20, cv_folds=5)

    # 4. Display Benchmark Results
    print("\n" + "=" * 70)
    print("5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARKS")
    print("=" * 70)
    cv_b = metadata["cv_benchmarks"]
    print(f"{'Model':<22} | {'ROC-AUC':<16} | {'PR-AUC':<16} | {'F1-Score':<16}")
    print("-" * 70)
    for m_name, metrics in cv_b.items():
        name_display = m_name.replace("_", " ").title()
        roc = f"{metrics['cv_roc_auc_mean']:.4f} (+/-{metrics['cv_roc_auc_std']:.3f})"
        pr = f"{metrics['cv_pr_auc_mean']:.4f} (+/-{metrics['cv_pr_auc_std']:.3f})"
        f1 = f"{metrics['cv_f1_mean']:.4f} (+/-{metrics['cv_f1_std']:.3f})"
        print(f"{name_display:<22} | {roc:<16} | {pr:<16} | {f1:<16}")

    print("\n" + "=" * 70)
    print("HOLDOUT TEST SET PERFORMANCE (362 Unseen Historical Tickets)")
    print("=" * 70)
    h_res = metadata["holdout_results"]
    for m_key in ["majority_baseline", "logistic_regression", "random_forest", "calibrated_selected_model"]:
        m_eval = h_res.get(m_key)
        if not m_eval:
            continue
        display_name = m_eval.get("model_name", m_key.replace("_", " ").title())
        cm = m_eval["confusion_matrix"]
        print(f"\nModel: {display_name}")
        print(f"  ROC-AUC:   {m_eval['roc_auc']:.4f}")
        print(f"  PR-AUC:    {m_eval['pr_auc']:.4f}")
        print(f"  F1-Score:  {m_eval['f1']:.4f}")
        print(f"  Precision: {m_eval['precision']:.4f}")
        print(f"  Recall:    {m_eval['recall']:.4f}")
        print(f"  Brier:     {m_eval['brier_score']:.4f}")
        print(f"  Confusion: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")

    # 5. Persist Artifacts
    print("\n[5/5] Saving model pipeline and evaluation metadata...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save()
    print(f"  - Model artifact: {model.DEFAULT_MODEL_PATH}")
    print(f"  - Metadata file:  {model.DEFAULT_METADATA_PATH}")

    # Top Feature Importances
    print("\nTop 5 Predictive Features (Pre-Assignment):")
    for item in metadata["top_feature_importances"][:5]:
        print(f"  - {item['feature']}: {item['importance']:.4f}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Model training and evaluation successfully completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
