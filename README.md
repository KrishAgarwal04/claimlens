# ClaimLens

**AI-assisted insurance claims document review dashboard that flags inconsistencies for human reviewers, never declaring a claim fraudulent.**

> ⚠️ **Status:** Frontend prototype. All data is mock data; there is no backend yet. See [Roadmap](#roadmap).

---

## Problem

Insurance companies receive claim forms, invoices, bills and supporting documents that are reviewed manually. This is slow and can miss inconsistencies. ClaimLens is the reviewer-facing dashboard for an AI-powered claims document processor that:

- extracts information from PDF, DOCX and image documents (OCR),
- compares claim details across documents and finds mismatches,
- validates extracted data against predefined policy rules,
- estimates the proportion of AI-generated content and highlights suspicious passages,
- shows every flag with a **confidence score and supporting evidence**,
- produces a downloadable report.

> **Core principle:** The system flags items *for human review*. It never automatically labels a claim as fraudulent. The final decision always rests with a human reviewer.

## Features (current frontend)

| Area | What it does |
|---|---|
| **Dashboard** | KPI cards, claims-over-time, flags by category, attention-level distribution, top triggered rules, recent claims with filters |
| **Upload Claim** | Drag & drop (PDF, DOCX, JPG, PNG, max 25 MB each), file-type validation, document-type selector, claim details form, analysis-scope toggles (OCR, cross-document check, policy rules, optional AI-content detection) |
| **Processing** | Step-by-step progress view with cancel and auto-redirect to the review page |
| **Claim Review** | 3-pane layout: document viewer with bounding-box overlays, extracted fields with OCR confidence and inline correction, and a findings panel |
| **Findings panel** | Summary, cross-document mismatches (side-by-side), policy rule results (pass/fail/N-A), AI-content estimate with caveats, other anomalies, audit trail and notes |
| **Reviewer actions** | Confirm issue, dismiss as false positive, comment, correct a field value, mark as reviewed |
| **Two-way linking** | Click a box on the document to highlight the finding, and click a finding to jump to the document location |
| **Claims Queue** | All claims sorted by attention level |
| **Reports** | Preview, print/save as PDF, download JSON and CSV |
| **Policy Rules** | Rule table with enable toggles and a live "test rule" panel |
| **Evaluation** | Precision, recall, F1, confusion matrix and sample test cases (demo values) |
| **Settings** | Theme, reviewer name, confidence threshold |

**UX details:** light/dark theme (light by default), keyboard shortcuts (`J`/`K` next/previous finding, `C` confirm, `D` dismiss, `Ctrl+K` search, `?` help, `Esc` close), neutral non-accusatory wording, and a persistent human-review disclaimer.

## Getting started

No build step or dependencies are needed.

```bash
git clone https://github.com/<your-username>/claimlens.git
cd claimlens
# open index.html in any modern browser
```

Or simply double-click `index.html`.

## Project structure

```
claimlens/
├── index.html       # page markup
├── style.css        # all styles (light/dark themes)
├── script.js        # mock data, state, views, router, actions
└── README.md
```

`script.js` is organised in sections: helpers, mock data, state, SVG charts, views, router and actions.

## Mock data

The prototype ships with 7 sample claims covering different scenarios:

| Claim | Scenario |
|---|---|
| CLM-2041 | Claim amount mismatch between claim form and invoice |
| CLM-2042 | Claim filed outside the 30-day policy window |
| CLM-2043 | High estimated AI-generated content in a medical narrative |
| CLM-2044 | Duplicate invoice number and missing signature |
| CLM-2045 | Clean claim, no flags |
| CLM-2046 | Low-quality scan with low OCR confidence |
| CLM-2047 | Still processing |

## Planned API contract

The mock layer is designed to be swapped for a real backend. Suggested endpoints:

```
POST   /api/claims/upload                 multipart files + metadata  ->  { claimId }
GET    /api/claims/:id/status             ->  { step, progress, state }
GET    /api/claims/:id                    ->  full claim object
GET    /api/claims                        ->  list with filters + pagination
PATCH  /api/claims/:id/findings/:fid      { status, comment }
PATCH  /api/claims/:id/fields/:fid        { correctedValue }
GET    /api/claims/:id/report?format=pdf|json|csv
GET    /api/rules
GET    /api/evaluation
```

Core data shape:

```
Claim   { id, name, pol, status, docs[], findings[], policy[], ai, summary }
Finding { id, cat, sev, title, why, conf, ev[], doc, box, status, comments[] }
Field   { id, label, val, conf, st, x, y, w, h }   // bbox in % of the page
```

## Roadmap

- [ ] Python backend (FastAPI) for upload and processing
- [ ] OCR and document extraction (PDF, DOCX, images)
- [ ] Cross-document comparison engine
- [ ] Rule engine for policy validation
- [ ] NLP/ML for AI-generated-content estimation and anomaly detection
- [ ] Real PDF/image viewer with server-provided bounding boxes
- [ ] Report generation (PDF) on the backend
- [ ] Evaluation on real sample documents
- [ ] Port frontend to React + Vite + TypeScript

## Limitations

- Frontend only, with mock data and simulated processing.
- Uploaded files are not actually read or analysed.
- AI-content detection is probabilistic and can produce false positives; the UI always labels it as an estimate.
- Evaluation page figures are demo values, not real benchmarks.

## Team

| Role | Responsibility |
|---|---|
| ML & Data | Dataset cleaning, forecasting, risk model, evaluation metrics |
| Dashboard & UI | Frontend interface, KPI cards, charts, filters, risk table |
| Recommendation Engine | Priority rules, workload optimisation, explanations, impact calculation |
| Integration & Testing | Module integration, API connections, end-to-end testing, deployment, demo |

## Disclaimer

ClaimLens is an assistive tool. Flags are prioritisation aids, not verdicts. All claim decisions must be made by a qualified human reviewer.

## License

MIT. Add a `LICENSE` file before publishing.
