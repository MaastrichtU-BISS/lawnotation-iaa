# IAA — Inter-Annotator Agreement for Span Annotations

A zero-dependency Go tool that measures how consistently multiple annotators mark up text spans across a shared document set. Designed for use with [LawNotation](https://lawnotation.org) exports, but compatible with any dataset that follows the same JSON schema.

---

## Background

When multiple human annotators label the same documents, measuring how much they agree is essential for validating annotation quality and label definitions. For **span annotations** (text segments tagged with a label), this is non-trivial: unlike categorical labels, spans have boundaries that annotators may shift by a word or two while still semantically agreeing. A single agreement number is also rarely sufficient — different research questions require different notions of "agreement."

This tool reports **two complementary views** per label, each answering a different question.

---

## The Two Views

### View 1 — Span Matching
> *"If one annotator is the reference, how accurately can another reproduce their spans?"*

Each annotator is treated as the reference in turn, and their spans are matched 1-to-1 using maximum bipartite matching against the other annotator's spans using the chosen `--criterion`. This finds the assignment that maximises true positives globally, avoiding the underestimation that greedy first-fit produces with overlapping or nested spans. Unmatched spans count as false positives or missed recalls.

**Metrics:** Precision, Recall, F1 — per pair in both directions, plus macro-average per label.

**Key property:** **Not chance-corrected** and **asymmetric** (precision/recall depend on who is the reference). Best suited for scenarios where one annotator is a gold standard, or for comparing annotation styles directionally. This is the dominant standard in NLP annotation papers (NER, relation extraction, legal NLP).

---

### View 2 — Coverage Agreement
> *"How much text do annotators agree on?"*

Each character or word position across all documents is treated as one item in a binary reliability matrix:
- `1` — this annotator's span covers this position for this label
- `0` — assigned to the document, but no span here
- `None` — annotator was not assigned to this document

**Metrics:** Krippendorff's α and Cohen's κ (all pairs).

**Key property:** This view is **length-weighted** — agreeing on a 200-character span contributes far more items than agreeing on a 5-character span. Use it to answer "how much text, in total, is consistently annotated?"

---

## Matching Criterion

The `--criterion` option controls when two spans are considered a match. It only affects View 1.

| Criterion | Behaviour |
|-----------|-----------|
| `exact` | Two spans must have identical `(start, end)` |
| `contained` | One span must be a subset of the other: `[a,b] ⊆ [c,d]` or vice versa |

> **Note:** `--criterion` has no effect on View 2. For coverage, a token is counted as covered if any span fully contains it — this is always containment-based by construction.

---

## Input Format

The tool reads a JSON file exported from LawNotation. The expected top-level structure is:

```json
{
  "labelset": {
    "labels": [
      { "name": "Actors", "color": "#71c345" },
      { "name": "Acts",   "color": "#4435ba" }
    ]
  },
  "documents": [
    {
      "name": "doc_001.txt",
      "full_text": "The full document text as a plain string.",
      "assignments": [
        {
          "annotator": 1,
          "status": "done",
          "annotations": [
            {
              "start": 4,
              "end": 59,
              "label": "Actors",
              "text": "the annotated substring"
            }
          ]
        }
      ]
    }
  ]
}
```

- `start` / `end` are **character offsets** (0-indexed, half-open: `text[start:end]`).
- An annotator can be assigned to a subset of documents; unassigned annotators are represented as `None` (not `0`) in the reliability matrix.
- Multiple annotators can be assigned to the same document.

---

## Usage

```bash
go run main.go iaa.go server.go --input <path/to/data.json> [options]
```

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--input` | path | *(required)* | Path to the LawNotation JSON export |
| `--criterion` | `exact` \| `contained` | `exact` | Match condition for View 1 |
| `--granularity` | `char` \| `word` | `word` | Token unit for View 2 |
| `--output` | path | `iaa_report.zip` | Output path for the ZIP report |

### Examples

```bash
# Word-level coverage, lenient (containment) matching
go run main.go iaa.go server.go --input input/data.json --criterion contained --granularity word --output output/report.zip

# Character-level coverage (slower, finer-grained)
go run main.go iaa.go server.go --input input/data.json --granularity char --output output/report-char.zip
```

To build a standalone binary instead of using `go run`:

```bash
go build -o iaa main.go iaa.go server.go
./iaa --input input/data.json --criterion contained --output output/report.zip
```

### Running as a server

Instead of the one-off CLI batch above, `--serve` starts an HTTP server that
computes metrics on demand for a web client:

```bash
go run main.go iaa.go server.go --serve --port 8080
```

or with the standalone binary:

```bash
go build -o iaa main.go iaa.go server.go
./iaa --serve --port 8080
```

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--serve` | flag | `false` | Start the HTTP server instead of running the CLI batch |
| `--port` | port number | `8080` | Port to listen on in `--serve` mode |

Endpoints (request body is the LawNotation JSON export, same schema as `--input`):

| Endpoint | Query params | Response |
|----------|--------------|----------|
| `POST /metrics` | `criterion`, `granularity` (same values/defaults as above) | JSON: `{ "annotation_metrics": <report>, "confidence_metrics": <difficulty rating summary> }` |
| `POST /report.zip` | `criterion`, `granularity` | The same ZIP report the CLI produces |

```bash
curl -X POST "http://localhost:8080/metrics?criterion=contained&granularity=word" \
  --data-binary @input/data.json

curl -X POST "http://localhost:8080/report.zip?criterion=contained&granularity=word" \
  --data-binary @input/data.json -o report.zip
```

#### Authentication

If the `IAA_API_KEY` environment variable is set, both endpoints require an
`Authorization: Bearer <key>` header; requests without it get `401
Unauthorized`. If `IAA_API_KEY` is unset, the server runs with no auth at all
(useful for local testing).

```bash
curl -X POST "http://localhost:8080/metrics" \
  -H "Authorization: Bearer $IAA_API_KEY" \
  --data-binary @input/data.json
```

There is still no CORS handling — this server is meant to be called
server-to-server (e.g. from another backend's server-side code), not
directly from browser JS.

---

## Output

The tool prints a summary table to stdout and writes a structured JSON report.

**Stdout example:**
```
=== SUMMARY ===
  Documents                          : 42
  Annotators                         : 1, 2, 3
  Granularity (coverage view)        : word
  Span matching       - mean F1       : 0.8103
  Coverage agreement  - mean α        : 0.7821
  Coverage agreement  - mean κ         : 0.7634

Label                              F1       cov α    cov κ
-----------------------------------------------------------------
Actors                             0.8450   0.8234   0.8101
Acts                               0.7900   0.7654   0.7500
...
```

**ZIP contents:**
```
report.zip
├── aggregate.json                        # Full aggregated report (JSON)
└── documents/
    ├── _aggregated/                      # Metrics and annotations across all documents
    │   └── <label>/
    │       ├── metrics.csv              # Aggregate span matching + coverage for this label
    │       └── annotations.csv          # All annotations for this label (+ document column)
    └── <document_name>/                  # File extension stripped (e.g. doc.txt → doc)
        └── <label>/
            ├── metrics.csv              # Span matching + coverage metrics for this label
            └── annotations.csv          # All annotations for this label by each annotator
```

**metrics.csv** columns: `pair, direction, tp, ref_count, sys_count, precision, recall, f1` (span matching section) followed by `metric, value` (coverage section with Krippendorff α and Cohen κ).

**annotations.csv** columns: `annotator, start, end, text` (per-document) or `document, annotator, start, end, text` (aggregated).

### Score interpretation guide (α and κ)

| Range | Interpretation |
|-------|----------------|
| < 0.20 | Slight agreement |
| 0.21 – 0.40 | Fair agreement |
| 0.41 – 0.60 | Moderate agreement |
| 0.61 – 0.80 | Substantial agreement |
| > 0.80 | Almost perfect agreement |

---

## Dependencies

None. Standard library only (`archive/zip`, `encoding/csv`, `encoding/json`, `flag`, `math`, `os`, `regexp`, `sort`, `strings`). Requires Go 1.18+.

---