"""Robust data loading, schema validation, and partitioning for OpsPilot AI.

Guarantees data integrity, referential validity, temporal sequencing,
and strict isolation between historical completed tickets and live open intake queue.
"""

from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    TICKETS_FILE,
    AGENTS_FILE,
    MERCHANTS_FILE,
    EXPECTED_ROW_COUNTS,
)


class DataValidationError(Exception):
    """Raised when operational datasets fail schema or referential integrity checks."""
    pass


class DataLoader:
    """Loads, validates, and partitions raw ITSM operational datasets."""

    MERCHANT_SCHEMA = {
        "merchant_id": "int64",
        "merchant_name": "object",
        "sector": "object",
        "tier": "object",
        "region": "object",
    }

    AGENT_SCHEMA = {
        "agent_id": "int64",
        "agent_name": "object",
        "tier": "object",
        "primary_category": "object",
        "shift_region": "object",
        "efficiency_multiplier": "float64",
    }

    TICKET_COLUMNS = [
        "ticket_id",
        "merchant_id",
        "category",
        "sub_category",
        "priority",
        "created_at",
        "is_legacy",
        "assigned_agent_id",
        "category_mismatch",
        "first_response_at",
        "closed_at",
        "ttfr_hours",
        "resolution_hours",
        "response_breached",
        "resolution_breached",
        "is_reopened",
        "is_reopen_child",
        "is_incident_ticket",
        "csat_score",
    ]

    def __init__(
        self,
        tickets_path: Optional[Path] = None,
        agents_path: Optional[Path] = None,
        merchants_path: Optional[Path] = None,
    ):
        self.tickets_path = Path(tickets_path or TICKETS_FILE)
        self.agents_path = Path(agents_path or AGENTS_FILE)
        self.merchants_path = Path(merchants_path or MERCHANTS_FILE)

    def load_merchants(self) -> pd.DataFrame:
        """Load merchants dataset and validate schema, nulls, and unique IDs.
        
        Returns:
            pd.DataFrame: Validated merchants dataset (112 records expected).
        """
        if not self.merchants_path.exists():
            raise FileNotFoundError(f"Merchants dataset not found at {self.merchants_path}")

        df = pd.read_csv(self.merchants_path)

        # 1. Row count validation
        if len(df) != EXPECTED_ROW_COUNTS["merchants"]:
            raise DataValidationError(
                f"Expected {EXPECTED_ROW_COUNTS['merchants']} merchants, found {len(df)}"
            )

        # 2. Column existence validation
        missing_cols = set(self.MERCHANT_SCHEMA.keys()) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing merchant columns: {missing_cols}")

        # 3. Unique ID validation
        if df["merchant_id"].duplicated().any():
            dupes = df[df["merchant_id"].duplicated()]["merchant_id"].tolist()
            raise DataValidationError(f"Duplicate merchant IDs found: {dupes}")

        # 4. Null value validation (merchants should have 0 nulls)
        null_counts = df.isnull().sum()
        if null_counts.any():
            raise DataValidationError(f"Unexpected null values in merchants: {null_counts.to_dict()}")

        return df

    def load_agents(self) -> pd.DataFrame:
        """Load agents dataset and validate schema, nulls, and unique IDs.
        
        Returns:
            pd.DataFrame: Validated agents dataset (20 records expected).
        """
        if not self.agents_path.exists():
            raise FileNotFoundError(f"Agents dataset not found at {self.agents_path}")

        df = pd.read_csv(self.agents_path)

        # 1. Row count validation
        if len(df) != EXPECTED_ROW_COUNTS["agents"]:
            raise DataValidationError(
                f"Expected {EXPECTED_ROW_COUNTS['agents']} agents, found {len(df)}"
            )

        # 2. Column existence validation
        missing_cols = set(self.AGENT_SCHEMA.keys()) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing agent columns: {missing_cols}")

        # 3. Unique ID validation
        if df["agent_id"].duplicated().any():
            dupes = df[df["agent_id"].duplicated()]["agent_id"].tolist()
            raise DataValidationError(f"Duplicate agent IDs found: {dupes}")

        # 4. Null value validation (agents should have 0 nulls)
        null_counts = df.isnull().sum()
        if null_counts.any():
            raise DataValidationError(f"Unexpected null values in agents: {null_counts.to_dict()}")

        # 5. Efficiency multiplier range check
        if (df["efficiency_multiplier"] <= 0).any():
            raise DataValidationError("Agent efficiency multiplier must be positive")

        return df

    def load_tickets(
        self,
        merchants_df: Optional[pd.DataFrame] = None,
        agents_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Load tickets dataset, parse timestamps, and enforce referential integrity.
        
        Returns:
            pd.DataFrame: Validated tickets dataset (2,057 records expected).
        """
        if not self.tickets_path.exists():
            raise FileNotFoundError(f"Tickets dataset not found at {self.tickets_path}")

        df = pd.read_csv(self.tickets_path)

        # 1. Row count validation
        if len(df) != EXPECTED_ROW_COUNTS["tickets"]:
            raise DataValidationError(
                f"Expected {EXPECTED_ROW_COUNTS['tickets']} tickets, found {len(df)}"
            )

        # 2. Column existence validation
        missing_cols = set(self.TICKET_COLUMNS) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing ticket columns: {missing_cols}")

        # 3. Unique Ticket ID validation
        if df["ticket_id"].duplicated().any():
            dupes = df[df["ticket_id"].duplicated()]["ticket_id"].tolist()
            raise DataValidationError(f"Duplicate ticket IDs found: {dupes}")

        # 4. Parse Timestamps
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["first_response_at"] = pd.to_datetime(df["first_response_at"])
        df["closed_at"] = pd.to_datetime(df["closed_at"])

        # 5. Referential Integrity Check - Merchants
        if merchants_df is None and self.merchants_path.exists():
            merchants_df = self.load_merchants()

        if merchants_df is not None:
            valid_merchant_ids = set(merchants_df["merchant_id"])
            invalid_merchants = set(df["merchant_id"]) - valid_merchant_ids
            if invalid_merchants:
                raise DataValidationError(
                    f"Foreign key violation: {len(invalid_merchants)} merchant IDs not found in merchants.csv"
                )

        # 6. Referential Integrity Check - Agents (for assigned historical tickets)
        if agents_df is None and self.agents_path.exists():
            agents_df = self.load_agents()

        if agents_df is not None:
            valid_agent_ids = set(agents_df["agent_id"])
            assigned_ticket_agents = set(df["assigned_agent_id"].dropna().astype(int))
            invalid_agents = assigned_ticket_agents - valid_agent_ids
            if invalid_agents:
                raise DataValidationError(
                    f"Foreign key violation: {len(invalid_agents)} agent IDs not found in agents.csv"
                )

        # 7. Temporal Sequence Validation (for completed historical tickets)
        legacy_df = df[df["is_legacy"] == True]
        invalid_resp_time = legacy_df[legacy_df["first_response_at"] < legacy_df["created_at"]]
        if not invalid_resp_time.empty:
            raise DataValidationError(
                f"{len(invalid_resp_time)} tickets have first_response_at earlier than created_at"
            )

        invalid_close_time = legacy_df[legacy_df["closed_at"] < legacy_df["first_response_at"]]
        if not invalid_close_time.empty:
            raise DataValidationError(
                f"{len(invalid_close_time)} tickets have closed_at earlier than first_response_at"
            )

        return df

    def split_tickets(
        self, tickets_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Partition tickets into historical records and live open queue with invariant checks.
        
        Args:
            tickets_df: Validated full tickets DataFrame.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - historical_df: Completed tickets (is_legacy == True, 1807 rows)
                - open_queue_df: Live incoming unassigned queue (is_legacy == False, 250 rows)
        """
        historical_df = tickets_df[tickets_df["is_legacy"] == True].copy()
        open_queue_df = tickets_df[tickets_df["is_legacy"] == False].copy()

        # Invariant 1: Historical row count
        if len(historical_df) != EXPECTED_ROW_COUNTS["historical_tickets"]:
            raise DataValidationError(
                f"Expected {EXPECTED_ROW_COUNTS['historical_tickets']} historical tickets, got {len(historical_df)}"
            )

        # Invariant 2: Open queue row count
        if len(open_queue_df) != EXPECTED_ROW_COUNTS["open_queue_tickets"]:
            raise DataValidationError(
                f"Expected {EXPECTED_ROW_COUNTS['open_queue_tickets']} open queue tickets, got {len(open_queue_df)}"
            )

        # Invariant 3: Open tickets must be completely unassigned and unhandled
        unassigned_cols = [
            "assigned_agent_id",
            "category_mismatch",
            "first_response_at",
            "closed_at",
            "ttfr_hours",
            "resolution_hours",
            "csat_score",
        ]
        for col in unassigned_cols:
            if not open_queue_df[col].isnull().all():
                non_null_count = open_queue_df[col].notnull().sum()
                raise DataValidationError(
                    f"Open queue tickets must have NULL {col}, found {non_null_count} non-null values"
                )

        # Invariant 4: Historical tickets must have complete operational metrics (except sparse csat_score)
        completed_cols = [
            "assigned_agent_id",
            "category_mismatch",
            "first_response_at",
            "closed_at",
            "ttfr_hours",
            "resolution_hours",
        ]
        for col in completed_cols:
            if historical_df[col].isnull().any():
                null_count = historical_df[col].isnull().sum()
                raise DataValidationError(
                    f"Historical tickets must have complete {col}, found {null_count} nulls"
                )

        # Invariant 5: Open ticket breach labels are unobserved placeholders
        # Add an explicit attribute / flag indicating open queue status
        open_queue_df["is_breach_unobserved_placeholder"] = True
        historical_df["is_breach_unobserved_placeholder"] = False

        return historical_df, open_queue_df

    def get_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and validate all three operational datasets in one call.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (tickets, agents, merchants)
        """
        merchants = self.load_merchants()
        agents = self.load_agents()
        tickets = self.load_tickets(merchants_df=merchants, agents_df=agents)
        return tickets, agents, merchants

    def run_diagnostics(self) -> Dict[str, Any]:
        """Generate a complete diagnostic quality report of the operational data.
        
        Returns:
            Dict[str, Any]: Verification status, distributions, nulls, and integrity checks.
        """
        tickets, agents, merchants = self.get_datasets()
        historical, open_queue = self.split_tickets(tickets)

        return {
            "status": "PASS",
            "counts": {
                "merchants": len(merchants),
                "agents": len(agents),
                "tickets_total": len(tickets),
                "historical_tickets": len(historical),
                "open_queue_tickets": len(open_queue),
            },
            "referential_integrity": {
                "unmatched_merchants": 0,
                "unmatched_agents": 0,
            },
            "temporal_boundaries": {
                "historical_start": str(historical["created_at"].min()),
                "historical_end": str(historical["created_at"].max()),
                "open_queue_start": str(open_queue["created_at"].min()),
                "open_queue_end": str(open_queue["created_at"].max()),
            },
            "historical_breach_rates": {
                "resolution_breached_pct": round(historical["resolution_breached"].mean() * 100, 2),
                "response_breached_pct": round(historical["response_breached"].mean() * 100, 2),
            },
            "category_mismatch": {
                "historical_mismatch_rate_pct": round(historical["category_mismatch"].mean() * 100, 2),
                "open_queue_null_count": int(open_queue["category_mismatch"].isnull().sum()),
            },
            "csat_survey_response_rate_pct": round(historical["csat_score"].notnull().mean() * 100, 2),
            "open_queue_placeholder_warning": (
                "Open tickets contain resolution_breached=False and response_breached=False "
                "as unobserved placeholders. Never use open tickets for model training."
            ),
        }
