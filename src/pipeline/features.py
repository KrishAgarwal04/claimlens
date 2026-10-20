"""Leakage-free feature engineering for OpsPilot AI.

Enforces strict temporal and operational boundaries to build pre-assignment
feature matrices using ONLY information available at initial ticket intake.
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.config import (
    INTAKE_RAW_COLUMNS,
    MERCHANT_CONTEXT_COLUMNS,
    FORBIDDEN_PRE_ASSIGNMENT_COLUMNS,
    PRE_ASSIGNMENT_FEATURE_COLUMNS,
    SLA_TARGETS,
)


class DataLeakageError(Exception):
    """Raised when forbidden outcome or post-intake features leak into feature matrices."""
    pass


class FeatureEngineer:
    """Extracts, joins, and encodes pre-assignment features from ticket and merchant records."""

    PRIORITY_MAP = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

    def __init__(self):
        self.feature_columns = [
            c for c in PRE_ASSIGNMENT_FEATURE_COLUMNS if c != "ticket_id"
        ]

    def validate_no_leakage(self, df: pd.DataFrame) -> None:
        """Verify that no outcome, response, or post-intake assignment features exist in df.
        
        Args:
            df: Feature DataFrame to validate.
            
        Raises:
            DataLeakageError: If any forbidden feature is detected.
        """
        leaked_cols = [c for c in FORBIDDEN_PRE_ASSIGNMENT_COLUMNS if c in df.columns]
        if leaked_cols:
            raise DataLeakageError(
                f"CRITICAL TARGET LEAKAGE: Forbidden columns found in feature matrix: {leaked_cols}. "
                "These features are post-intake outcomes or require agent assignment, violating pre-assignment boundaries."
            )

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive cyclical and calendar features strictly from ticket intake timestamp.
        
        Args:
            df: DataFrame containing 'created_at'.
            
        Returns:
            pd.DataFrame: Augmented with temporal attributes.
        """
        out = df.copy()
        dt = out["created_at"].dt

        out["intake_hour"] = dt.hour
        out["intake_day_of_week"] = dt.dayofweek
        out["intake_is_weekend"] = out["intake_day_of_week"].isin([5, 6]).astype(int)
        out["intake_month"] = dt.month

        # Cyclical transforms (24h clock and 7d week)
        out["intake_hour_sin"] = np.sin(2 * np.pi * out["intake_hour"] / 24.0)
        out["intake_hour_cos"] = np.cos(2 * np.pi * out["intake_hour"] / 24.0)
        out["intake_dow_sin"] = np.sin(2 * np.pi * out["intake_day_of_week"] / 7.0)
        out["intake_dow_cos"] = np.cos(2 * np.pi * out["intake_day_of_week"] / 7.0)

        return out

    def merge_merchant_context(
        self, tickets_df: pd.DataFrame, merchants_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Join merchant metadata available in system prior to ticket creation.
        
        Args:
            tickets_df: Tickets DataFrame.
            merchants_df: Merchants reference DataFrame.
            
        Returns:
            pd.DataFrame: Tickets with merged merchant attributes.
        """
        merchant_subset = merchants_df[
            ["merchant_id", "tier", "sector", "region"]
        ].copy()
        merchant_subset.rename(
            columns={
                "tier": "merchant_tier",
                "sector": "merchant_sector",
                "region": "merchant_region",
            },
            inplace=True,
        )

        merged = tickets_df.merge(merchant_subset, on="merchant_id", how="left")
        if merged["merchant_tier"].isnull().any():
            missing_count = merged["merchant_tier"].isnull().sum()
            raise ValueError(f"Failed to join merchant context for {missing_count} tickets")

        return merged

    def build_pre_assignment_features(
        self, tickets_df: pd.DataFrame, merchants_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Construct pure pre-assignment feature matrix from tickets and merchants.
        
        Args:
            tickets_df: Raw or partition ticket DataFrame.
            merchants_df: Validated merchants DataFrame.
            
        Returns:
            pd.DataFrame: Feature matrix containing only pre-assignment attributes.
        """
        # 1. Merge merchant context
        df = self.merge_merchant_context(tickets_df, merchants_df)

        # 2. Extract temporal features from intake timestamp
        df = self.extract_temporal_features(df)

        # 3. Derive priority rank and SLA target hours (defaulting missing priority to P3 baseline)
        df["priority_rank"] = df["priority"].map(self.PRIORITY_MAP).fillna(3).astype(int)
        df["sla_target_hours"] = df["priority"].map(
            lambda p: SLA_TARGETS.get(p, {}).get("resolution_hours", 24.0)
        ).fillna(24.0)

        # 4. Standardize intake boolean flags to 0/1 integers
        df["is_reopen_child"] = df["is_reopen_child"].fillna(False).astype(int)
        df["is_incident_ticket"] = df["is_incident_ticket"].fillna(False).astype(int)

        # 5. Filter strictly to standardized pre-assignment columns
        X = df[PRE_ASSIGNMENT_FEATURE_COLUMNS].copy()

        # 6. Verify zero target leakage
        self.validate_no_leakage(X)

        return X

    def prepare_training_dataset(
        self, historical_tickets: pd.DataFrame, merchants_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepares leak-free training matrix X and target y from historical completed tickets.
        
        Guarantees:
            1. Raises ValueError if open-queue tickets (is_legacy == False) are passed.
            2. Target y is binary (1 = breached, 0 = within SLA).
            3. Feature matrix X contains strictly pre-assignment features.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_features, y_target)
        """
        # Enforce that no open tickets are used in training
        if not (historical_tickets["is_legacy"] == True).all():
            open_count = (historical_tickets["is_legacy"] == False).sum()
            raise ValueError(
                f"Cannot train on open queue tickets: detected {open_count} open tickets. "
                "Open tickets carry placeholder breach labels and unclosed statuses."
            )

        X = self.build_pre_assignment_features(historical_tickets, merchants_df)
        y = historical_tickets["resolution_breached"].astype(int).reset_index(drop=True)

        # Maintain index alignment
        X = X.reset_index(drop=True)

        return X, y

    def prepare_open_queue_dataset(
        self, open_tickets: pd.DataFrame, merchants_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepares pre-assignment feature matrix for live unassigned open tickets.
        
        Guarantees:
            1. Validates that tickets are indeed open queue tickets (is_legacy == False).
            2. Never returns placeholder breach flags as valid labels.
            3. Feature matrix schema perfectly matches training features.
        
        Returns:
            pd.DataFrame: X_open feature matrix ready for SLA risk inference.
        """
        if not (open_tickets["is_legacy"] == False).all():
            legacy_count = (open_tickets["is_legacy"] == True).sum()
            raise ValueError(
                f"Expected open queue tickets only, found {legacy_count} historical tickets"
            )

        X_open = self.build_pre_assignment_features(open_tickets, merchants_df)
        return X_open.reset_index(drop=True)

    @classmethod
    def get_feature_metadata(cls) -> List[Dict[str, Any]]:
        """Detailed documentation explaining each pre-assignment feature and its intake validity."""
        return [
            {
                "column": "ticket_id",
                "dtype": "int64",
                "category": "Identifier",
                "intake_justification": "Primary key generated by ticketing intake portal upon record creation.",
            },
            {
                "column": "category",
                "dtype": "object",
                "category": "Ticket Metadata",
                "intake_justification": "Selected by merchant / customer during initial helpdesk submission.",
            },
            {
                "column": "sub_category",
                "dtype": "object",
                "category": "Ticket Metadata",
                "intake_justification": "Selected by merchant / customer specifying root issue at submission.",
            },
            {
                "column": "priority",
                "dtype": "object",
                "category": "Service Tier",
                "intake_justification": "Assigned at creation (P1-P4) based on merchant tier and reported impact.",
            },
            {
                "column": "priority_rank",
                "dtype": "int64",
                "category": "Service Tier",
                "intake_justification": "Ordinal mapping (P1=1, P2=2, P3=3, P4=4) derived deterministically from priority.",
            },
            {
                "column": "sla_target_hours",
                "dtype": "float64",
                "category": "Service Tier",
                "intake_justification": "Contractual SLA threshold (2h, 8h, 24h, 48h) mapped from ticket priority.",
            },
            {
                "column": "is_reopen_child",
                "dtype": "int64",
                "category": "Ticket Metadata",
                "intake_justification": "Flag indicating follow-up escalation ticket spawned at intake.",
            },
            {
                "column": "is_incident_ticket",
                "dtype": "int64",
                "category": "System Context",
                "intake_justification": "Platform incident correlation flag present at intake.",
            },
            {
                "column": "intake_hour",
                "dtype": "int64",
                "category": "Temporal",
                "intake_justification": "Hour of day (0-23) extracted from ticket intake timestamp created_at.",
            },
            {
                "column": "intake_day_of_week",
                "dtype": "int64",
                "category": "Temporal",
                "intake_justification": "Day of week (0=Mon, 6=Sun) extracted from created_at.",
            },
            {
                "column": "intake_is_weekend",
                "dtype": "int64",
                "category": "Temporal",
                "intake_justification": "Binary weekend flag indicating weekend staffing shift.",
            },
            {
                "column": "intake_month",
                "dtype": "int64",
                "category": "Temporal",
                "intake_justification": "Month (1-12) extracted from created_at.",
            },
            {
                "column": "intake_hour_sin",
                "dtype": "float64",
                "category": "Temporal Cyclical",
                "intake_justification": "Sine cyclical encoding of 24h intake hour.",
            },
            {
                "column": "intake_hour_cos",
                "dtype": "float64",
                "category": "Temporal Cyclical",
                "intake_justification": "Cosine cyclical encoding of 24h intake hour.",
            },
            {
                "column": "intake_dow_sin",
                "dtype": "float64",
                "category": "Temporal Cyclical",
                "intake_justification": "Sine cyclical encoding of 7-day intake week.",
            },
            {
                "column": "intake_dow_cos",
                "dtype": "float64",
                "category": "Temporal Cyclical",
                "intake_justification": "Cosine cyclical encoding of 7-day intake week.",
            },
            {
                "column": "merchant_id",
                "dtype": "int64",
                "category": "Merchant Profile",
                "intake_justification": "Merchant identifier submitted with the ticket request.",
            },
            {
                "column": "merchant_tier",
                "dtype": "object",
                "category": "Merchant Profile",
                "intake_justification": "Account tier (Foundation or Advanced) pre-existing in CRM profile.",
            },
            {
                "column": "merchant_sector",
                "dtype": "object",
                "category": "Merchant Profile",
                "intake_justification": "Merchant industry sector pre-existing in CRM profile.",
            },
            {
                "column": "merchant_region",
                "dtype": "object",
                "category": "Merchant Profile",
                "intake_justification": "Merchant operating region pre-existing in CRM profile.",
            },
        ]
