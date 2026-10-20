# OpsPilot AI: Dataset Dictionary & Verification Report

## 1. Verified Summary Table

| Dataset | Row Count | Column Count | Duplicate IDs | Null Count Summary | Key ID Column |
|---|---|---|---|---|---|
| `merchants.csv` | 112 | 5 | 0 | 0 nulls across all columns | `merchant_id` |
| `agents.csv` | 20 | 6 | 0 | 0 nulls across all columns | `agent_id` |
| `tickets.csv` | 2,057 | 19 | 0 | 250 in open queue cols, 1,832 in `csat_score` | `ticket_id` |

---

## 2. Table Schemas & Distributions

### A. `merchants.csv` (112 rows)
- `merchant_id` (int64, PK): Unique range 101 to 212.
- `merchant_name` (object): Company name.
- `sector` (object):
  - Fashion: 23
  - Grocery: 22
  - Sporting Goods: 19
  - Home & Garden: 18
  - Pharmaceuticals: 16
  - Electronics: 14
- `tier` (object):
  - Foundation: 90 (80.4%)
  - Advanced: 22 (19.6%)
- `region` (object):
  - UK & Europe: 40 (35.7%)
  - Asia-Pacific: 38 (33.9%)
  - North America: 34 (30.4%)

### B. `agents.csv` (20 rows)
- `agent_id` (int64, PK): Unique range 201 to 220.
- `agent_name` (object): Agent full name.
- `tier` (object):
  - L1: 10 agents (50%)
  - L2: 7 agents (35%)
  - L3: 3 agents (15%)
- `primary_category` (object):
  - Account Access: 5
  - Fulfilment & Logistics: 5
  - API Integrations: 4
  - Payments & Checkout: 4
  - Notifications: 2
- `shift_region` (object):
  - Asia-Pacific: 11
  - North America: 7
  - UK & Europe: 2
- `efficiency_multiplier` (float64):
  - Min: 0.692, 25%: 0.849, Median: 1.073, Mean: 1.044, Max: 1.355.

### C. `tickets.csv` (2,057 rows)
- `ticket_id` (int64, PK): 2,057 unique IDs.
- `merchant_id` (int64, FK): 100% matched to `merchants.csv`.
- `category` (object): 5 primary domains.
- `sub_category` (object): 20 subcategories.
- `priority` (object): P1 (112), P2 (400), P3 (1013), P4 (532).
- `created_at` (datetime): Range 2025-01-02 01:47:43 to 2026-06-30 23:59:36.
- `is_legacy` (bool): Historical partition flag (1,807 True, 250 False).
- `assigned_agent_id` (float64, FK): Missing 250 (all in open queue).
- `category_mismatch` (bool/object): Missing 250 (all in open queue).
  - Legacy True: 538 (29.8%), Legacy False: 1,269 (70.2%).
- `first_response_at` (datetime/object): Missing 250 (all in open queue).
- `closed_at` (datetime/object): Missing 250 (all in open queue).
- `ttfr_hours` (float64): Missing 250 (all in open queue).
- `resolution_hours` (float64): Missing 250 (all in open queue).
- `response_breached` (bool): 345 True in legacy (19.1%), False for all 250 open tickets.
- `resolution_breached` (bool): 1,037 True in legacy (57.4%), False for all 250 open tickets.
- `is_reopened` (bool): 307 True, 1,750 False.
- `is_reopen_child` (bool): 307 True, 1,750 False (mutually exclusive with `is_reopened`).
- `is_incident_ticket` (bool): 2,057 False (0 True in dataset).
- `csat_score` (float64): Present for 225 tickets (12.5% of legacy tickets), 1,832 missing.

---

## 3. SLA Targets by Priority (Empirically Derived)

| Priority | Time to First Response (TTFR) Target | Resolution Time Target | Historical TTFR Breach Rate | Historical Resolution Breach Rate |
|---|---|---|---|---|
| **P1** | 0.25 hours (15 min) | 2.0 hours (120 min) | 66.7% | 71.9% |
| **P2** | 1.00 hour (60 min) | 8.0 hours | 57.1% | 74.6% |
| **P3** | 4.00 hours | 24.0 hours (1 day) | 7.4% | 52.8% |
| **P4** | 12.00 hours | 48.0 hours (2 days) | 3.0% | 50.1% |

---

## 4. Key Empirical Insights & Anomalies

1. **Category Mismatch Impact**:
   - When `category_mismatch == False`: Resolution breach rate is **49.96%** (avg 28.2 hours).
   - When `category_mismatch == True`: Resolution breach rate spikes to **74.91%** (avg 40.4 hours), a **+24.95 percentage point penalty**.
   - 29.8% of historical tickets had category mismatches, demonstrating huge potential for OpsPilot's recommendation engine.

2. **Agent Tier Performance**:
   - **L1 Agents**: 75.7% resolution breach rate, 36.5h average resolution time.
   - **L2 Agents**: 46.8% resolution breach rate, 30.4h average resolution time.
   - **L3 Agents**: 35.6% resolution breach rate, 23.2h average resolution time.

3. **Open Queue State**:
   - All 250 tickets created between `2026-06-29 00:25:03` and `2026-06-30 23:59:36` represent unhandled tickets requiring triage and intelligent routing.
