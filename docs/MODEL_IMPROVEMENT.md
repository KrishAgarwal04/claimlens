# Model accuracy comparison

Candidate rejected; the original model remains active at 80.39% holdout accuracy and 73.67% separate-evaluation accuracy.

Model and threshold selected using training OOF predictions only; fixed evaluation sets used for assessment and deployment gate.

Selected candidate: `gradient_depth1_trees250`; decision threshold: 0.525. Configurations tested: 60.

| Dataset | Baseline accuracy | Candidate accuracy | Candidate precision | Candidate recall | Candidate ROC-AUC | Paired p-value |
|---|---:|---:|---:|---:|---:|---:|
| historical_holdout (n=362) | 80.39% | 80.39% | 81.00% | 86.06% | 0.8833 | 1.0000 |
| evaluation_tickets (n=300) | 73.67% | 72.67% | 70.05% | 83.44% | 0.8409 | 0.3750 |
| synthetic_canonical (n=500) | 72.80% | 73.80% | 73.63% | 82.37% | 0.8074 | 0.4049 |

Candidate promoted: False.

All datasets are synthetic; accuracy is not a real-world performance guarantee.
Evaluation sets existed and earlier results were known before this assessment; use a new untouched dataset for independent confirmation.
Selection CV scores have selection bias; holdout scores are reported separately.

A paired p-value >= 0.05 does not establish a statistically significant accuracy difference on that dataset.
