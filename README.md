# OpsPilot AI - Business Operations Intelligence & Decision Engine

For the current technology stack, architecture, verified model results, demo script,
limitations, and 70 judges' questions with answers, including detailed Random Forest
and other algorithm explanations, see the
[Complete Project and Judges' Preparation Guide](docs/PROJECT_AND_JUDGES_GUIDE.md).

The prototype now includes two SLA prediction stages, capacity-constrained dispatch,
evidence for individual recommendations, workload forecasts, surge detection and
baseline-versus-proposed scenario comparisons.

| Prediction stage | Original holdout (362 tickets) | Separate evaluation (300 tickets) | V2 evaluation (500 tickets) |
|---|---:|---:|---:|
| Arrival only: agent unknown | 80.39% accuracy | 73.67% accuracy | 79.00% accuracy |
| Compact assignment model: proposed agent known | **85.36% accuracy** | **87.33% accuracy** | **88.80% accuracy** |

The active assignment model is strongly regularized logistic regression (C=0.01),
selected near 85% training-CV accuracy to satisfy the requested compact profile.
Its fixed decision threshold is 0.5. This deliberately trades predictive performance
for a simpler model; it is not an accuracy improvement over the previous full profile.
The previous stronger artifact is retained under `models/baseline/` and its
benchmarks are archived in `docs/*_high_accuracy.*`.

The assignment-time model uses 26 intake and proposed-agent attributes. It does
not use response or resolution outcomes as features. These stages use different
available information; all datasets are synthetic. See [the evaluation report](EVALUATION_REPORT.md)
and [the verified assignment metrics](docs/assignment_accuracy_results.json).

Open **AI Decision Engine**, the default dashboard view, to edit per-agent planning
budgets, review recommended assignments and export the dispatch plan. **Workload
Forecast** shows a backtest, forecast intervals and unusual intake days. Forecast
dates follow the dataset snapshot rather than today's date.

## Dataset

A dataset intended to be explored to surface business problems that may be solved in modelling projects or as recommendations in case studies

## What is in this repository

- merchants.csv, 112 merchant accounts
- agents.csv, 20 support agents
- tickets.csv, just over two thousand tickets spanning eighteen months of history plus a live open queue

## Schema

### merchants.csv

| Column | Description |
|---|---|
| merchant_id | Numeric identifier |
| merchant_name | Generated company name |
| sector | Industry sector |
| tier | Advanced or Foundation, reflects account volume |
| region | Operating region |

### agents.csv

| Column | Description |
|---|---|
| agent_id | Numeric identifier |
| agent_name | Agent name |
| tier | L1, L2 or L3 support level |
| primary_category | The ticket category this agent specialises in |
| shift_region | Region the agent's shift covers |
| efficiency_multiplier | A persistent per agent performance factor that shapes response and resolution time |

### tickets.csv

| Column | Description |
|---|---|
| ticket_id | Numeric identifier |
| merchant_id | Foreign key to merchants |
| category, sub_category | Ticket classification |
| priority | P1 to P4 |
| created_at | Ticket creation timestamp |
| is_legacy | Whether the ticket belongs to the historical batch rather than the live open queue |
| assigned_agent_id | Foreign key to agents, blank for new tickets still awaiting routing |
| category_mismatch | Whether the assigned agent's specialty differs from the ticket category |
| first_response_at, closed_at | Response and resolution timestamps |
| ttfr_hours, resolution_hours | Duration metrics in hours |
| response_breached, resolution_breached | Breach flags against SLA targets tied to priority |
| is_reopened | A probability driven outcome, built entirely from intake time features |
| is_reopen_child | Whether the record is the escalated follow up spawned by a reopen |
| is_incident_ticket | Whether the ticket was generated or affected by one of three simulated platform incidents |
| csat_score | Present for roughly one in eight closed tickets, reflecting realistic survey non response |

New tickets carry blank values in several columns, including first_response_at, closed_at and csat_score. This is by design, since these tickets have not been processed yet, not a sign of missing or broken data.

## How to use it

Works in whatever tool you use. Load the CSVs into Python, R or a SQL database and query, model or visualise from there. No dependency on any particular language or platform.

## Project ideas

A ticket routing model is the one currently in progress. Comparing how well tickets are handled before and after automated routing is a natural next step once that exists. Predictive modelling of customer satisfaction is a further direction, useful in its own right and as something that could feed into the routing work.

## Known issues and feedback

This is the first iteration of my first synthetic dataset. I would be grateful for any feedback that identifies problems or suggestions

## Run the OpsPilot AI prototype

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Evaluate the saved model without retraining:

```powershell
.\.venv\Scripts\python.exe -m src.models.evaluate
.\.venv\Scripts\python.exe -m src.models.evaluate_assignment
.\.venv\Scripts\python.exe -m src.models.evaluate_dataset --dataset evaluation_tickets_v2.csv --name evaluation_v2
```

Retrain the assignment-time model using training-only cross-validation:

```powershell
.\.venv\Scripts\python.exe -m src.models.train_assignment --profile compact
```

The assignment artifact is separate from the arrival-only model. Proposed agent
attributes are known before work starts; observed mismatch flags and post-response
or post-resolution outcomes never enter its feature matrix. The planner marks
already-overdue open tickets using the snapshot deadline, and does not pretend
that reassignment can undo an existing breach. Agent budgets are explicit planning
assumptions. Predicted scenario gains need operational confirmation.

For arrival-only model experiments, install the optional training dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m src.models.improvement
.\.venv\Scripts\python.exe -m src.models.refine
.\.venv\Scripts\python.exe -m src.models.optimize
```

Selection uses five-fold cross-validation on the original 1,445 training tickets.
The refinement step compares leading models and support-vector classifiers over
three repeat splits of the training set.
The 362-ticket holdout and the separate evaluation datasets do not fit models or
select thresholds. The promotion command requires better accuracy on both the
original holdout and the separate 300-ticket evaluation dataset, and retains the
original artifact under `models/baseline/` locally. Restart Streamlit after promotion.

See `docs/MODEL_IMPROVEMENT.md` and `docs/model_improvement_results.json` for the
measured comparison, uncertainty and limitations. These are synthetic datasets;
fresh real operational data is needed to validate deployment performance.
