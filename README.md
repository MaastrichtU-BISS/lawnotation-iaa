# IAA — Inter-Annotator Agreement for Span Annotations

A zero-dependency Go tool that measures how consistently multiple annotators mark up text spans across a shared document set. Designed for use with [LawNotation](https://lawnotation.org) exports, but compatible with any dataset that follows the same JSON schema.

---

## Background

When multiple human annotators label the same documents, measuring how much they agree is essential for validating annotation quality and label definitions. For **span annotations** (text segments tagged with a label), this is non-trivial: unlike categorical labels, spans have boundaries that annotators may shift by a word or two while still semantically agreeing. A single agreement number is also rarely sufficient — different research questions require different notions of "agreement."

This tool reports **two complementary views** per label, each answering a different question.

---

## The Two Views

### View 1 — Span Matching
> *"Where do two annotators' spans agree, and how do they differ where they don't?"*

For every pair of annotators, spans are matched 1-to-1 in a single pass: the
greedy matcher prefers the largest overlap, then identical bounds, then the
same label, with a full tie-break so the result does not depend on the order
annotations arrive in. Each span then falls into one of the categories of
Ortmann's (2022) error taxonomy:

| Category | Meaning |
|----------|---------|
| matched | same text, same label |
| boundary differs | same label, overlapping text with different bounds (smaller, larger or shifted) |
| label differs | same text, different label |
| both differ | overlapping text, different bounds and label |
| only one saw it | one annotator marked it, the other marked nothing there |

**Metrics:** strict and lenient F1 per pair, read from that one matching pass.
Strict gives a partial match no credit — it counts as a spurious span and a
missed one. Lenient counts it as half of each. The gap between the two is how
much of the score rests on partial matches. F1 is symmetric, so each pair is
reported once.

Per label, a label error cannot occur by construction (both sides hold only
that label), so the per-label profiles show only matched, boundary and
only-one categories.

**Key property:** **not chance-corrected.** This is the dominant standard in
NLP annotation papers (NER, relation extraction, legal NLP).

---

### View 2 — Coverage Agreement
> *"How much text do annotators agree on?"*

Each word position across all documents is treated as one item in a binary reliability matrix:
- `1` — this annotator's span covers this position for this label
- `0` — assigned to the document, but no span here
- `None` — annotator was not assigned to this document

**Metrics:** Krippendorff's α over all annotators, α and Cohen's κ for every
pair, and α per document (for the documents where anyone used the label).

**Key property:** This view is **length-weighted** — agreeing on a 40-word span contributes far more items than agreeing on a 2-word span — and words nobody marked count as agreement. Use it to answer "how much text, in total, is consistently annotated?"

### Document labels

On a document-level task, and for the second labelset of a hybrid one, a label
is a tag on a whole document. There is nothing to match and no extent, so only
View 2 applies, with one item per document: did each annotator apply the label
to it. There is no per-document α — at a single item α is always 0 or 1.

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
go run . --input <path/to/data.json> [options]
```

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--input` | path | *(required)* | Path to the LawNotation JSON export |
| `--output` | path | `iaa_report.zip` | Output path for the ZIP report |

There are no scoring options: strict and lenient F1 are both always reported,
and coverage is always counted in words.

```bash
go run . --input input/data.json --output output/report.zip
```

To build a standalone binary instead of using `go run`:

```bash
go build -o iaa .
./iaa --input input/data.json --output output/report.zip
```

### Running as a server

Instead of the one-off CLI batch above, `--serve` starts an HTTP server that
computes metrics on demand for a web client:

```bash
go run . --serve --port 8080
```

or with the standalone binary:

```bash
go build -o iaa .
./iaa --serve --port 8080
```

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--serve` | flag | `false` | Start the HTTP server instead of running the CLI batch |
| `--port` | port number | `8080` | Port to listen on in `--serve` mode |

Endpoints (request body is the LawNotation JSON export, same schema as `--input`):

| Endpoint | Response |
|----------|----------|
| `POST /metrics` | JSON: `{ "annotation_metrics": <report>, "confidence_metrics": <confidence rating summary> }` |
| `POST /report.zip` | The same ZIP report the CLI produces |

```bash
curl -X POST "http://localhost:8080/metrics" --data-binary @input/data.json

curl -X POST "http://localhost:8080/report.zip" --data-binary @input/data.json -o report.zip
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

### Embedding in another Go program

The same routes are available as a mountable `http.Handler`, so a host
application can serve them in-process rather than running this as a separate
service. Requests are then same-origin, which sidesteps the CORS gap above.

```go
import "github.com/MaastrichtU-BISS/lawnotation-iaa/iaa"

mux.Handle("/api/services/lawnotation-iaa/", http.StripPrefix(
    "/api/services/lawnotation-iaa", iaa.Handler()))
```

The mounted paths are `metrics` and `report.zip`, relative to the mount point.

---

## Output

The tool prints a summary table to stdout and writes a ZIP of a few tidy CSV
tables, one row per observation.

**Stdout example:**
```
=== SUMMARY ===
  Documents        : 900
  Annotators       : 1, 2, 3
  Annotation level : span

Label                              Kind         alpha  F1 strict F1 lenient
---------------------------------------------------------------------------
Postcondition                      span        0.8818     0.6670     0.7675
Precondition                       span        0.7666     0.6287     0.6662
...
```

`F1 strict` and `F1 lenient` are the mean over annotator pairs.

**ZIP contents:**
```
report.zip
├── README.md            # What each table holds, column by column
├── labels.csv           # One row per label: α, mean strict/lenient F1, how much was marked
├── pairs.csv            # One row per label and annotator pair: α, κ, both F1s, the error profile
├── documents.csv        # One row per label and document: α per document
├── document_pairs.csv   # One row per label, document and pair: both F1s and the error profile there
├── annotations.csv      # Every annotation, with document and annotator
├── assignments.csv      # Every assignment: its confidence rating and annotation count
├── confidence.csv       # Confidence ratings per annotator, and an "all" row
└── confidence_pairs.csv # α on the confidence ratings for every pair
```

The `README.md` inside the archive is generated from the same table
definitions as the files, so its column list always matches them. An empty
cell means the value does not exist for that row, never zero. The web client
(vue-iaa-metrics) adds a `report.pdf` with the results page to the same
archive.

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

None. Standard library only. Requires Go 1.23+.

---

## Hybrid tasks (span + document labels)

With `"annotation_level": "hybrid"` a task has two labelsets: `labelset` holds the span labels and `document_labelset` the whole-document labels. Each assignment carries both `annotations` (spans, as above) and `document_annotations` (`[{ "label": "..." }]`).

- Span labels are reported exactly as in a span-level task (`per_label`).
- Document labels are reported as presence agreement (Krippendorff's α, Cohen's κ) under `per_document_label`; there is no span matching for them. A label name may appear in both labelsets.
- In the ZIP, `labels.csv` has a row for each, told apart by its `kind` column (`span` or `document`), and `annotations.csv` lists the document tags with kind `document` and no position.
- Confidence metrics are unchanged.
