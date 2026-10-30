# Model accuracy comparison

Model and threshold selected using training OOF predictions only; fixed evaluation sets used once for assessment and deployment gate.

Selected model: `gradient_depth1_trees250`; decision threshold: 0.550.

| Dataset | Before accuracy | After accuracy | After precision | After recall | After ROC-AUC | Paired p-value |
|---|---:|---:|---:|---:|---:|---:|
| historical_holdout (n=362) | 80.39% | 80.39% | 81.00% | 86.06% | 0.8833 | 1.0000 |
| evaluation_tickets (n=300) | 73.67% | 72.67% | 70.05% | 83.44% | 0.8409 | 0.3750 |
| synthetic_canonical (n=500) | 72.80% | 73.80% | 73.63% | 82.37% | 0.8074 | 0.4049 |

Candidate promoted: False.

All datasets are synthetic; accuracy is not a real-world performance guarantee.
Evaluation sets existed and baseline results were known before this work; use a new untouched dataset for independent confirmation.
Selection CV scores have selection bias; holdout scores are reported separately.
