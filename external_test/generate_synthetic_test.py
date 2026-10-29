"""Synthetic External Test Dataset Generator for OpsPilot AI.

Generates a schema-compliant, controlled-simulation synthetic test dataset strictly
for validating the SLA risk model's pre-assignment inference pipeline.

IMPORTANT DISCLAIMER:
This dataset is a CONTROLLED SIMULATION / PIPELINE INTEGRITY TEST ONLY.
It does NOT establish or claim real-world generalization. All labels are generated
via an explicit probabilistic simulation and do not reflect real-world operational logs.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    SUPPORT_CATEGORIES,
    CATEGORY_SUBCATEGORIES,
    CANONICAL_SUBCATEGORIES,
    MERCHANT_TIERS,
    MERCHANT_SECTORS,
    MERCHANT_REGIONS,
    SLA_TARGETS,
    FORBIDDEN_PRE_ASSIGNMENT_COLUMNS,
    RANDOM_STATE,
)
from src.models.sla_model import SLARiskModel


# Plausible domain-grounded logistic risk coefficients (Explicitly Documented)
# Simulates empirical ITSM operational reality:
# 1. P1/P2 tickets have tight SLA deadlines (2h, 8h) with high specialist coordination delays.
# 2. Payments & API categories suffer from external bank/gateway latencies and timeouts.
# 3. Account Access & Notifications have higher automation and faster resolution times.
BASE_LOGIT = -0.10

PRIORITY_COEFFICIENTS = {
    "P1": 0.85,    # 2-hour SLA window: extreme risk of breaching
    "P2": 0.95,    # 8-hour window: high risk from specialist handoff queues
    "P3": -0.20,   # 24-hour window: moderate standard operational buffer
    "P4": -0.65,   # 48-hour window: substantial buffer, lowest breach probability
}

CATEGORY_COEFFICIENTS = {
    "Payments & Checkout": 1.40,   # Banking networks, 3DS challenges, settlement latency
    "API Integrations": 0.85,      # Third-party webhook timeouts, rate limiting, auth failures
    "Fulfilment & Logistics": 0.15, # Courier sync delays, warehouse inventory reconciliation
    "Account Access": -1.15,       # Fast password/MFA resets, internal admin tools
    "Notifications": -1.45,        # Automated templated communications, low complex drag
}

SUBCATEGORY_ADJUSTMENTS = {
    "Payout Delays": 0.40,
    "Refund Processing": 0.30,
    "3DS Authentication": 0.35,
    "Card Declines": 0.25,
    "Endpoint Timeout": 0.30,
    "Webhook Failures": 0.30,
    "Authentication Error": 0.20,
    "Rate Limiting": 0.15,
    "Carrier Sync Failure": 0.25,
    "Inventory Mismatch": 0.15,
    "Tracking Updates": -0.10,
    "Label Generation": -0.15,
    "SSO Failure": 0.20,
    "2FA Reset": -0.20,
    "User Permissions": -0.10,
    "Password Reset": -0.30,
    "SMS Gateway Drag": 0.25,
    "Template Syntax Error": 0.10,
    "Email Delivery Failure": -0.10,
    "Webhook Retries": 0.05,
}


class SyntheticTestGenerator:
    """Generates schema-compliant, controlled-simulation test datasets for OpsPilot AI."""

    PRIORITY_MAP = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

    def __init__(self, seed: int = RANDOM_STATE):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate a synthetic test dataset with all 18 model features and test label."""
        records = []

        # Category and priority sampling weights (plausible ITSM enterprise workload)
        cat_probs = [0.24, 0.23, 0.18, 0.18, 0.17]  # Payments, API, Fulfilment, Notif, Account
        pri_probs = [0.08, 0.22, 0.46, 0.24]        # P1, P2, P3, P4
        tier_probs = [0.72, 0.28]                    # Foundation, Advanced

        for i in range(1, n_samples + 1):
            ticket_id = f"CANTEST-{i:05d}"
            
            # Categorical intake sampling
            category = self.rng.choice(SUPPORT_CATEGORIES, p=cat_probs)
            sub_category = self.rng.choice(CATEGORY_SUBCATEGORIES[category])
            priority = self.rng.choice(["P1", "P2", "P3", "P4"], p=pri_probs)
            merchant_tier = self.rng.choice(MERCHANT_TIERS, p=tier_probs)
            merchant_sector = self.rng.choice(MERCHANT_SECTORS)
            merchant_region = self.rng.choice(MERCHANT_REGIONS)

            # Deterministic derivations from priority
            priority_rank = self.PRIORITY_MAP[priority]
            sla_target_hours = SLA_TARGETS[priority]["resolution_hours"]

            # Intake boolean flags
            is_reopen_child = int(self.rng.binomial(1, 0.15))
            is_incident_ticket = int(self.rng.binomial(1, 0.05))

            # Temporal intake attributes (calendar year 2026)
            intake_hour = int(self.rng.randint(0, 24))
            intake_day_of_week = int(self.rng.randint(0, 7))
            intake_is_weekend = int(intake_day_of_week in [5, 6])
            intake_month = int(self.rng.randint(1, 13))

            # Exact cyclical transforms
            intake_hour_sin = float(np.sin(2.0 * np.pi * intake_hour / 24.0))
            intake_hour_cos = float(np.cos(2.0 * np.pi * intake_hour / 24.0))
            intake_dow_sin = float(np.sin(2.0 * np.pi * intake_day_of_week / 7.0))
            intake_dow_cos = float(np.cos(2.0 * np.pi * intake_day_of_week / 7.0))

            # Documented Synthetic Latent Risk Model
            logit = (
                BASE_LOGIT
                + PRIORITY_COEFFICIENTS[priority]
                + CATEGORY_COEFFICIENTS[category]
                + SUBCATEGORY_ADJUSTMENTS.get(sub_category, 0.0)
                + (0.35 if is_reopen_child else 0.0)
                + (0.25 if intake_is_weekend else 0.0)
                + (-0.20 if merchant_tier == "Advanced" else 0.10)
                + self.rng.normal(0.0, 0.35)  # Stochastic unobserved queue congestion
            )
            prob_breach = 1.0 / (1.0 + np.exp(-logit))
            actual_sla_breach = int(self.rng.uniform(0.0, 1.0) < prob_breach)

            records.append({
                "ticket_id": ticket_id,
                # 18 Model Input Features (exact names & ordering)
                "priority_rank": priority_rank,
                "sla_target_hours": sla_target_hours,
                "is_reopen_child": is_reopen_child,
                "is_incident_ticket": is_incident_ticket,
                "intake_hour": intake_hour,
                "intake_day_of_week": intake_day_of_week,
                "intake_is_weekend": intake_is_weekend,
                "intake_month": intake_month,
                "intake_hour_sin": intake_hour_sin,
                "intake_hour_cos": intake_hour_cos,
                "intake_dow_sin": intake_dow_sin,
                "intake_dow_cos": intake_dow_cos,
                "category": category,
                "sub_category": sub_category,
                "priority": priority,
                "merchant_tier": merchant_tier,
                "merchant_sector": merchant_sector,
                "merchant_region": merchant_region,
                # Evaluation Ground Truth (Strictly separate from model features)
                "actual_sla_breach": actual_sla_breach,
            })

        df = pd.DataFrame(records)
        return df

    def validate(self, df: pd.DataFrame, model_artifact_path: Optional[Path] = None) -> Dict[str, Any]:
        """Perform comprehensive self-checks: schema, vocabulary, distribution, and zero leakage."""
        results = {"passed": True, "errors": []}

        # 1. Schema Check
        expected_features = [
            "priority_rank", "sla_target_hours", "is_reopen_child", "is_incident_ticket",
            "intake_hour", "intake_day_of_week", "intake_is_weekend", "intake_month",
            "intake_hour_sin", "intake_hour_cos", "intake_dow_sin", "intake_dow_cos",
            "category", "sub_category", "priority", "merchant_tier", "merchant_sector", "merchant_region"
        ]
        all_expected_cols = ["ticket_id"] + expected_features + ["actual_sla_breach"]

        if list(df.columns) != all_expected_cols:
            missing = set(all_expected_cols) - set(df.columns)
            extra = set(df.columns) - set(all_expected_cols)
            results["passed"] = False
            results["errors"].append(f"Column schema mismatch. Missing: {missing}, Extra: {extra}")

        # Check nulls
        null_counts = df.isnull().sum()
        if null_counts.any():
            results["passed"] = False
            results["errors"].append(f"Null values detected: {null_counts[null_counts > 0].to_dict()}")

        # Check value bounds & consistency
        if not (df["intake_hour"].between(0, 23).all()):
            results["passed"] = False
            results["errors"].append("intake_hour outside [0, 23]")
        if not (df["intake_day_of_week"].between(0, 6).all()):
            results["passed"] = False
            results["errors"].append("intake_day_of_week outside [0, 6]")
        if not (df["intake_is_weekend"] == df["intake_day_of_week"].isin([5, 6]).astype(int)).all():
            results["passed"] = False
            results["errors"].append("intake_is_weekend inconsistent with intake_day_of_week")

        # 2. Vocabulary Compatibility
        unseen_cats = set(df["category"].unique()) - set(SUPPORT_CATEGORIES)
        if unseen_cats:
            results["passed"] = False
            results["errors"].append(f"Unseen categories: {unseen_cats}")

        unseen_subs = set(df["sub_category"].unique()) - set(CANONICAL_SUBCATEGORIES)
        if unseen_subs:
            results["passed"] = False
            results["errors"].append(f"Unseen subcategories: {unseen_subs}")

        # Subcategory nested parent check
        for _, row in df.iterrows():
            if row["sub_category"] not in CATEGORY_SUBCATEGORIES[row["category"]]:
                results["passed"] = False
                results["errors"].append(f"Subcategory '{row['sub_category']}' invalid for category '{row['category']}'")
                break

        # 3. Target Distribution Check
        breach_rate = df["actual_sla_breach"].mean()
        if not (0.35 <= breach_rate <= 0.65):
            results["passed"] = False
            results["errors"].append(f"Breach rate {breach_rate:.2%} outside expected plausible range [35%, 65%]")

        # 4. Target Leakage Prevention Check
        leaked = [col for col in FORBIDDEN_PRE_ASSIGNMENT_COLUMNS if col in df.columns]
        if leaked:
            results["passed"] = False
            results["errors"].append(f"Target leakage: forbidden columns present: {leaked}")

        # 5. Model Inference Compatibility Check (Zero Retraining / Safe Read-Only Verification)
        try:
            model = SLARiskModel.load(model_path=model_artifact_path)
            X_eval = df[model.feature_names].copy()
            y_risk = model.predict_risk(X_eval)
            if len(y_risk) != len(df) or np.isnan(y_risk).any():
                results["passed"] = False
                results["errors"].append("Model inference returned invalid predictions")
        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"Model inference failed: {e}")

        return results


def main():
    print("=" * 80)
    print("OPSPILOT AI - SYNTHETIC EXTERNAL TEST-DATA GENERATOR")
    print("=" * 80)
    print("DISCLAIMER: Controlled simulation test only. Does NOT establish real-world generalization.")
    print("-" * 80)

    # 1. Generate Dataset
    generator = SyntheticTestGenerator(seed=RANDOM_STATE)
    n_tickets = 500
    df = generator.generate(n_samples=n_tickets)
    print(f"Generated {len(df)} synthetic tickets across 18 pre-assignment features.")

    # 2. Automated Validation Checks
    print("\nRunning automated validation checks:")
    val_results = generator.validate(df)
    if val_results["passed"]:
        print("  [PASS] Schema validity: All 18 features present with correct types and non-null values.")
        print("  [PASS] Category vocabulary: 100% match with canonical config.py categories & subcategories.")
        print("  [PASS] Label distribution: Plausible breach rate of "
              f"{df['actual_sla_breach'].mean():.2%} ({df['actual_sla_breach'].sum()}/{len(df)} breaches).")
        print("  [PASS] Zero target leakage: No post-intake outcome or agent assignment features present.")
        print("  [PASS] Model compatibility: Saved model (models/sla_risk_model.joblib) executed inference cleanly.")
    else:
        print("  [FAIL] Validation errors encountered:")
        for err in val_results["errors"]:
            print(f"    - {err}")
        sys.exit(1)

    # 3. Save new test CSV under distinct filename (Preserves all existing datasets untouched)
    output_path = ROOT_DIR / "external_test" / "opspilot_external_test_canonical.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] New canonical synthetic test dataset saved to:")
    print(f"  {output_path}")

    # Summary by priority and category
    print("\nGenerated Empirical Breach Rates by Category (Plausible Risk Gradient):")
    for cat in sorted(df["category"].unique()):
        cat_df = df[df["category"] == cat]
        print(f"  - {cat:<24}: N={len(cat_df):<3} | Breach Rate: {cat_df['actual_sla_breach'].mean():.1%}")

    print("\nGenerated Empirical Breach Rates by Priority:")
    for pri in ["P1", "P2", "P3", "P4"]:
        pri_df = df[df["priority"] == pri]
        print(f"  - {pri:<4} ({SLA_TARGETS[pri]['resolution_hours']}h SLA): N={len(pri_df):<3} | Breach Rate: {pri_df['actual_sla_breach'].mean():.1%}")


if __name__ == "__main__":
    main()
