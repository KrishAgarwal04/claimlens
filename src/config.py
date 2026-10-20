"""Configuration settings, schemas, and invariants for OpsPilot AI."""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

# Source Data Paths (Original CSVs must be preserved untouched)
TICKETS_FILE = DATA_DIR / "tickets.csv"
AGENTS_FILE = DATA_DIR / "agents.csv"
MERCHANTS_FILE = DATA_DIR / "merchants.csv"

# Expected Dataset Shapes & Invariants
EXPECTED_ROW_COUNTS = {
    "merchants": 112,
    "agents": 20,
    "tickets": 2057,
    "historical_tickets": 1807,
    "open_queue_tickets": 250,
}

# SLA Target Definitions (derived empirically from dataset)
SLA_TARGETS = {
    "P1": {"ttfr_hours": 0.25, "resolution_hours": 2.0},
    "P2": {"ttfr_hours": 1.00, "resolution_hours": 8.0},
    "P3": {"ttfr_hours": 4.00, "resolution_hours": 24.0},
    "P4": {"ttfr_hours": 12.00, "resolution_hours": 48.0},
}

# Operational Domain Categories
SUPPORT_CATEGORIES = [
    "Payments & Checkout",
    "API Integrations",
    "Notifications",
    "Fulfilment & Logistics",
    "Account Access",
]

# Random State
RANDOM_STATE = 42

# --- COLUMN CLASSIFICATIONS FOR LEAKAGE PREVENTION ---

# 1. Available strictly at ticket intake (before assignment)
INTAKE_RAW_COLUMNS = [
    "ticket_id",
    "merchant_id",
    "category",
    "sub_category",
    "priority",
    "created_at",
    "is_reopen_child",
    "is_incident_ticket",
]

# 2. Merchant context available at intake via merchant_id join
MERCHANT_CONTEXT_COLUMNS = [
    "merchant_tier",
    "merchant_sector",
    "merchant_region",
]

# 3. Post-intake assignment features (NOT available at intake; forbidden in pre-assignment triage)
ASSIGNMENT_COLUMNS = [
    "assigned_agent_id",
    "category_mismatch",
]

# 4. Agent metadata (NOT available at intake)
AGENT_METADATA_COLUMNS = [
    "agent_name",
    "agent_tier",
    "primary_category",
    "shift_region",
    "efficiency_multiplier",
]

# 5. Post-response operational features (NOT available at intake)
POST_RESPONSE_COLUMNS = [
    "first_response_at",
    "ttfr_hours",
    "response_breached",
]

# 6. Post-resolution outcome features (Strict target leakage; forbidden in predictive features)
OUTCOME_COLUMNS = [
    "closed_at",
    "resolution_hours",
    "resolution_breached",
    "csat_score",
    "is_reopened",
]

# Complete set of forbidden features for pre-assignment SLA prediction
FORBIDDEN_PRE_ASSIGNMENT_COLUMNS = (
    ASSIGNMENT_COLUMNS
    + AGENT_METADATA_COLUMNS
    + POST_RESPONSE_COLUMNS
    + OUTCOME_COLUMNS
)

# Backward-compatible alias
LEAKAGE_FEATURES = FORBIDDEN_PRE_ASSIGNMENT_COLUMNS


# Standardized pre-assignment engineered feature names
PRE_ASSIGNMENT_FEATURE_COLUMNS = [
    "ticket_id",
    "category",
    "sub_category",
    "priority",
    "priority_rank",
    "sla_target_hours",
    "is_reopen_child",
    "is_incident_ticket",
    "intake_hour",
    "intake_day_of_week",
    "intake_is_weekend",
    "intake_month",
    "intake_hour_sin",
    "intake_hour_cos",
    "intake_dow_sin",
    "intake_dow_cos",
    "merchant_id",
    "merchant_tier",
    "merchant_sector",
    "merchant_region",
]
