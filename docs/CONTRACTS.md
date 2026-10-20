# OpsPilot AI: Architecture Contracts

This document formalizes the input and output contracts, schemas, and invariants for each module within OpsPilot AI.

---

## 1. Data Pipeline Module (`src.pipeline`)

### Purpose
Ingests raw CSV files, validates schemas, enforces referential integrity, partitions historical from live open queues, and extracts leak-free features.

### DataLoader Contract (`loader.py`)
- **Inputs**:
  - `tickets_path: Path` (default: `tickets.csv`)
  - `agents_path: Path` (default: `agents.csv`)
  - `merchants_path: Path` (default: `merchants.csv`)
- **Outputs**:
  - `tickets_df: pd.DataFrame` (2,057 rows, 19 cols, parsed datetimes)
  - `agents_df: pd.DataFrame` (20 rows, 6 cols)
  - `merchants_df: pd.DataFrame` (112 rows, 5 cols)
  - `split_tickets(tickets_df)`:
    - `historical_df: pd.DataFrame` (`is_legacy == True`, 1,807 rows)
    - `open_queue_df: pd.DataFrame` (`is_legacy == False`, 250 rows)
- **Invariants**:
  - All foreign keys (`merchant_id`, `assigned_agent_id`) must referentially match.
  - No alteration to raw CSV data on disk.

### FeatureEngineer Contract (`features.py`)
- **Inputs**:
  - `historical_df: pd.DataFrame`, `merchants_df: pd.DataFrame`, `agents_df: Optional[pd.DataFrame]`
- **Outputs**:
  - `prepare_training_features()`:
    - `X: pd.DataFrame` (1,807 rows, feature columns strictly limited to intake/context)
    - `y: pd.Series` (1,807 rows, binary `resolution_breached`)
  - `prepare_inference_features(open_queue_df, merchants_df)`:
    - `X_open: pd.DataFrame` (250 rows, identical feature schema as `X`)
- **Invariants / Target Leakage Protections**:
  - Excluded from `X`: `resolution_hours`, `closed_at`, `ttfr_hours`, `first_response_at`, `response_breached`, `resolution_breached`, `csat_score`.
  - For pre-assignment triage, `assigned_agent_id` and `category_mismatch` are omitted from intake features.

---

## 2. SLA Risk Model Module (`src.models`)

### Purpose
Estimates the probability that an incoming ticket will breach its resolution SLA window, prioritizing high-risk cases for rapid triage and escalation.

### SLARiskModel Contract (`sla_model.py`)
- **Inputs**:
  - Training: `X: pd.DataFrame` (leak-free features), `y: pd.Series` (binary labels).
  - Inference: `X_open: pd.DataFrame` (250 open queue rows).
- **Outputs**:
  - `predict_proba(X) -> np.ndarray`: Array of float probabilities $[0.0, 1.0]$.
  - `predict(X, threshold=0.5) -> np.ndarray`: Binary classifications $\{0, 1\}$.
  - `evaluate(X_val, y_val) -> Dict[str, float]`:
    - `roc_auc: float`
    - `f1_score: float`
    - `precision: float`
    - `recall: float`
    - `brier_score: float`
- **Invariants**:
  - Validated using time-based or stratified k-fold cross-validation on `historical_df` only.
  - Probability calibration verified via Brier score / reliability curve.

---

## 3. Bottleneck Detection Module (`src.analytics`)

### Purpose
Diagnoses systematic operational inefficiencies across category specialties, agent tiers, regional shifts, and ticket complexities.

### BottleneckDetector Contract (`bottlenecks.py`)
- **Inputs**:
  - `tickets_df: pd.DataFrame` (historical tickets with completion records)
  - `agents_df: pd.DataFrame`
  - `merchants_df: pd.DataFrame`
- **Outputs**:
  - `detect_category_mismatches() -> Dict[str, Any]`:
    - `mismatch_rate: float` (historical % of mismatched tickets)
    - `matched_breach_rate: float` (~50.0%)
    - `mismatched_breach_rate: float` (~74.9%)
    - `breach_penalty: float` (~+24.9 pp)
    - `hours_penalty: float` (~+12.1 hours)
  - `detect_category_hotspots() -> pd.DataFrame`:
    - Columns: `[category, total_volume, p1_volume, breach_rate, avg_resolution_hours, mismatch_rate]`
  - `detect_agent_load_imbalance() -> pd.DataFrame`:
    - Columns: `[agent_id, agent_name, tier, primary_category, shift_region, ticket_count, breach_rate, efficiency_multiplier]`
- **Invariants**:
  - Metrics derived deterministically from historical data without synthetic fabrication.

---

## 4. Agent Recommendation Module (`src.routing`)

### Purpose
Ranks and recommends optimal support agents for incoming unassigned tickets to minimize breach risk, eliminate category mismatch, and balance queue load.

### AgentRecommender Contract (`recommender.py`)
- **Inputs**:
  - Single Ticket Triage: `ticket: Dict[str, Any]` (open ticket attributes: `category`, `sub_category`, `priority`, `merchant_region`).
  - Batch Queue Routing: `open_tickets_df: pd.DataFrame` (250 unassigned tickets).
  - System State: `agents_df: pd.DataFrame`, optional `current_agent_workload: Dict[int, int]`.
- **Outputs**:
  - `recommend_for_ticket()`:
    - Returns `List[Dict[str, Any]]` of top-k agents:
      - `agent_id: int`
      - `agent_name: str`
      - `tier: str`
      - `specialty_match: bool`
      - `region_match: bool`
      - `efficiency_multiplier: float`
      - `composite_score: float` (weighted match score)
      - `projected_breach_risk: float`
      - `recommendation_rationale: str`
  - `route_batch()`:
    - Returns `pd.DataFrame` (250 rows) with assigned `recommended_agent_id`, `recommended_agent_name`, `routing_tier`, and `expected_risk`.
- **Invariants**:
  - Strict preservation of agent capacity constraints (avoids overloading high-efficiency agents).
  - P1 tickets prioritized toward L2/L3 agents with matching primary categories.

---

## 5. Impact Simulator Module (`src.simulation`)

### Purpose
Provides interactive counterfactual what-if analysis enabling operations managers to simulate policy changes before deploying them to production.

### ImpactSimulator Contract (`simulator.py`)
- **Inputs**:
  - `target_mismatch_rate: float` (e.g. reduced from 29.8% to 5.0%)
  - `staffing_adjustments: Dict[str, int]` (e.g. +2 L2 agents to Payments)
  - `routing_policy: str` (Baseline First-Come-First-Serve vs OpsPilot Smart Routing)
- **Outputs**:
  - `simulation_results: Dict[str, Any]`:
    - `baseline_breach_rate: float` (57.4% historical)
    - `projected_breach_rate: float`
    - `breach_reduction_percentage: float`
    - `hours_saved_monthly: float`
    - `projected_cost_savings: float` (calculated based on configurable agent hourly cost and breach penalties)
- **Invariants**:
  - Grounded strictly in empirical coefficients derived from historical data (e.g., mismatch breach delta = 24.9 pp; L1 vs L3 breach delta = 40.1 pp).
  - Rejects impossible scenarios (e.g., negative mismatch rates, zero resolution times).

---

## 6. Streamlit Dashboard Contract (`app/`)

### Purpose
Executive and operational interface delivering real-time queue visibility, model explanations, bottleneck heatmaps, smart routing recommendations, and scenario simulations.

### Views / Page Contracts
1. **Executive Overview**: High-level KPIs, open queue counter, historical breach rates by priority, SLA target scorecard.
2. **SLA Risk Predictor**: Triage tool displaying 250 open tickets ranked by predicted breach probability, risk bucket filters (High, Medium, Low), and feature importance / risk drivers.
3. **Bottleneck Diagnostics**: Plotly heatmaps of category mismatches, agent tier distributions, and regional coverage mismatches.
4. **Smart Dispatcher**: Interactive ticket assignment sandbox with 1-click automated batch routing and single-ticket candidate inspector.
5. **ROI & Impact Simulator**: Interactive sliders for mismatch reduction and agent headcount shifts showing live recalculations of hours saved and breach reductions.
