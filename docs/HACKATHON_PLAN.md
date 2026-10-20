# OpsPilot AI: 8-Day Hackathon Implementation Plan

This roadmap outlines a realistic, sprint-based plan to develop, validate, and demonstrate OpsPilot AI within an 8-day hackathon window.

---

## Sprint Overview

```
Day 1: Ingestion & Feature Engineering (Zero Leakage)
Day 2: SLA Risk Model Training & Calibration
Day 3: Triage Scorer & Feature Attribution
Day 4: Operational Bottleneck Diagnostics Engine
Day 5: Smart Agent Recommendation & Batch Routing
Day 6: What-If Impact Simulator & Business ROI Modeling
Day 7: Full Streamlit Dashboard & Plotly Visualizations
Day 8: End-to-End Testing, Demo Rehearsal & Pitch Deck
```

---

## Day-by-Day Implementation Roadmap

### Day 1: Ingestion Pipeline & Feature Engineering
- **Focus**: Data contract validation and leakage-free feature pipelines.
- **Tasks**:
  1. Finalize `src/pipeline/loader.py` with schema asserts and foreign key validation.
  2. Implement `src/pipeline/features.py` with temporal cyclical features (`hour_of_day`, `day_of_week`), merchant embeddings/aggregates.
  3. Strict partition validation: ensure 1,807 historical training rows and 250 open queue inference rows are completely isolated.
  4. Unit test pipeline transformations in `tests/test_pipeline.py`.
- **Deliverable**: Clean `X_train`, `y_train`, and `X_open` matrices ready for model training.

---

### Day 2: SLA Risk Model Training & Calibration
- **Focus**: High-performance, calibrated binary classification for SLA resolution breaches.
- **Tasks**:
  1. Train and benchmark candidate algorithms in `src/models/sla_model.py`:
     - Baseline: Stratified Logistic Regression / Balanced Random Forest.
     - Primary: Histogram Gradient Boosting Classifier (`HistGradientBoostingClassifier` or LightGBM).
  2. Implement 5-fold Stratified / Temporal Cross-Validation.
  3. Evaluate metrics: ROC-AUC, PR-AUC, F1-Score, Brier Score.
  4. Perform probability calibration (Isotonic/Sigmoid) so predicted probabilities reflect true breach risks.
  5. Export serialized model artifact to `models/sla_risk_model.joblib`.
- **Deliverable**: Calibrated ML model with validated ROC-AUC and PR-AUC.

---

### Day 3: Explainability & Open Queue Triage Scorer
- **Focus**: Turning model predictions into actionable triage insights.
- **Tasks**:
  1. Implement global and local feature importance (SHAP or Tree feature importances) in `src/models/`.
  2. Build open queue scoring pipeline: score all 250 open tickets with breach probabilities.
  3. Define operational risk tiers:
     - 🔴 **High Risk** ($P(\text{breach}) \ge 0.70$)
     - 🟡 **Medium Risk** ($0.40 \le P(\text{breach}) < 0.70$)
     - 🟢 **Low Risk** ($P(\text{breach}) < 0.40$)
  4. Generate top 3 risk factors per ticket (e.g., "P1 priority + High intake hour + Historical merchant friction").
- **Deliverable**: Ranked open queue dataframe with risk scores and explanation rationale.

---

### Day 4: Bottleneck Analytics & Root Cause Engine
- **Focus**: Surface systemic failure modes in support operations.
- **Tasks**:
  1. Implement `src/analytics/bottlenecks.py`:
     - Category mismatch diagnosis: quantify the 24.9 pp breach rate penalty.
     - Agent tier efficiency analysis: analyze L1 (75.7% breach) vs L3 (35.6% breach).
     - Shift region coverage gaps (e.g., UK & Europe having only 2 agents for 40 merchants).
  2. Prepare Plotly-ready aggregation dataframes for interactive visual heatmaps.
- **Deliverable**: Automated diagnosis module providing quantitative operational bottleneck reports.

---

### Day 5: Smart Agent Recommendation & Routing Engine
- **Focus**: Algorithmic dispatcher matching tickets to the best available agent.
- **Tasks**:
  1. Implement `src/routing/recommender.py`:
     - Rule 1: Category Specialty Match (primary category == ticket category).
     - Rule 2: Tier Suitability (P1/P2 tickets routed to L2/L3 agents).
     - Rule 3: Efficiency & Regional Shift Matching.
     - Rule 4: Workload Balancer (prevents overloading top agents).
  2. Build batch router: process all 250 open tickets and generate recommended assignments.
  3. Validate that smart routing eliminates unnecessary category mismatches.
- **Deliverable**: Recommendation engine with single-ticket recommendations and batch routing capabilities.

---

### Day 6: What-If Impact Simulator & Business ROI
- **Focus**: Business justification and executive decision support.
- **Tasks**:
  1. Implement `src/simulation/simulator.py`:
     - Scenario A: "What if category mismatch is reduced to < 5%?"
     - Scenario B: "What if we add 2 L2 agents to Payments & Checkout?"
     - Scenario C: "What if automated dispatching is applied to the 250 open tickets?"
  2. Model hours saved, breach reduction percentage, and estimated financial impact ($ ROI).
- **Deliverable**: Configurable simulator generating before/after comparisons and executive metrics.

---

### Day 7: Streamlit Dashboard & Interactive UI
- **Focus**: Polished, intuitive user experience for judges and business operators.
- **Tasks**:
  1. Build multi-tab Streamlit interface in `app/main.py`:
     - Tab 1: **Executive Overview & Queue Health** (KPI cards, priority distributions).
     - Tab 2: **Live Queue & SLA Risk Triage** (sortable table, risk badges, SHAP factor inspector).
     - Tab 3: **Bottleneck Heatmaps** (interactive Plotly sankey/heatmaps).
     - Tab 4: **Agent Dispatcher Sandbox** (manual vs smart 1-click batch allocation).
     - Tab 5: **What-If Impact Simulator** (interactive sliders, dynamic ROI curves).
  2. Add custom styling, clean tooltips, and responsive layout.
- **Deliverable**: Fully functional, interactive Streamlit application.

---

### Day 8: Testing, Rehearsal, & Hackathon Pitch
- **Focus**: Demo reliability, story cohesion, and competition readiness.
- **Tasks**:
  1. Run end-to-end regression tests across all modules.
  2. Verify performance (instant Streamlit re-renders with caching).
  3. Create standard demo walkthrough script:
     - The Problem: 57.4% historical SLA breaches driven by 29.8% category mismatches.
     - The Solution: OpsPilot AI (Predict -> Diagnose -> Route -> Simulate).
     - The Impact: Measurable breach reduction and saved operational hours.
  4. Final code cleanup and documentation freeze.
- **Deliverable**: Rock-solid presentation deck, clean repository, and flawless live demo.
