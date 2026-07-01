# IAA — Inter-Annotator Agreement for Span Annotations

A zero-dependency Python tool that measures how consistently multiple annotators mark up text spans across a shared document set. Designed for use with [LawNotation](https://lawnotation.org) exports, but compatible with any dataset that follows the same JSON schema.

---

## Background

When multiple human annotators label the same documents, measuring how much they agree is essential for validating annotation quality and label definitions. For **span annotations** (text segments tagged with a label), this is non-trivial: unlike categorical labels, spans have boundaries that annotators may shift by a word or two while still semantically agreeing. A single agreement number is also rarely sufficient — different research questions require different notions of "agreement."

This tool reports **three complementary views** per label, each answering a different question.

---

## The Three Views

### View 1 — Coverage Agreement
> *"How much text do annotators agree on?"*

Each character or word position across all documents is treated as one item in a binary reliability matrix:
- `1` — this annotator's span covers this position for this label
- `0` — assigned to the document, but no span here
- `None` — annotator was not assigned to this document

**Metrics:** Krippendorff's α and Cohen's κ (all pairs).

**Key property:** This view is **length-weighted** — agreeing on a 200-character span contributes far more items than agreeing on a 5-character span. Use it to answer "how much text, in total, is consistently annotated?"

---

### View 2 — Decision Agreement
> *"How often do annotators make the same annotation decisions?"*

All spans from all annotators that touch or overlap each other for a given label are merged into a single **positive cluster**. Gaps between clusters become **negative clusters**. Each cluster, regardless of its character length, counts as exactly one item:
- `1` — this annotator has any span that matches the cluster (per `--criterion`)
- `0` — assigned but no matching span
- `None` — not assigned

**Metrics:** Krippendorff's α and Cohen's κ (all pairs).

**Key property:** **Length-neutral** — a 3-word cluster and a 300-word cluster both count as one decision. For each positive cluster, one negative item is added, keeping a strict **1:1 ratio** of annotated vs unannotated decisions per document. This avoids both inflation from too many trivial 0-0 items (sparse labels) and deflation from having no non-annotation baseline. Use it to answer "how consistently do annotators agree on annotation decisions, regardless of span length?"

---

### View 3 — Span Matching
> *"If one annotator is the reference, how accurately can another reproduce their spans?"*

Each annotator is treated as the reference in turn, and their spans are matched 1-to-1 (greedy, first-fit) against the other annotator's spans using the chosen `--criterion`. Unmatched spans count as false positives or missed recalls.

**Metrics:** Precision, Recall, F1 — per pair in both directions, plus macro-average per label.

**Key property:** **Not chance-corrected** and **asymmetric** (precision/recall depend on who is the reference). Best suited for scenarios where one annotator is a gold standard, or for comparing annotation styles directionally.

---

## Matching Criterion

The `--criterion` option controls when two spans (or a span and a cluster) are considered a match. It affects Views 2 and 3.

| Criterion | View 2 | View 3 |
|-----------|--------|--------|
| `exact` | Annotator's span must exactly match the cluster boundaries `(start, end)` | Two spans must have identical `(start, end)` |
| `contained` | Annotator's span must overlap the cluster at all | One span must be a subset of the other: `[a,b] ⊆ [c,d]` or vice versa |

> **Note:** `--criterion` has no effect on View 1. For coverage, a token is counted as covered if any span fully contains it — this is always containment-based by construction.

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
python iaa.py --input <path/to/data.json> [options]
```

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--input` | path | *(required)* | Path to the LawNotation JSON export |
| `--criterion` | `exact` \| `contained` | `exact` | Match condition for Views 2 and 3 |
| `--granularity` | `char` \| `word` | `word` | Token unit for View 1 |
| `--output` | path | `iaa_report.json` | Output path for the JSON report |

### Examples

```bash
# Word-level coverage, lenient (containment) matching
python iaa.py --input input/data.json --criterion contained --granularity word --output output/report-word-contained.json

# Character-level coverage (slower, finer-grained)
python iaa.py --input input/data.json --granularity char --output output/report-char.json
```

---

## Output

The tool prints a summary table to stdout and writes a structured JSON report.

**Stdout example:**
```
=== SUMMARY ===
  Documents                          : 42
  Annotators                         : annotator_1, annotator_2, annotator_3
  Granularity (coverage view)        : word
  Coverage agreement  - mean α       : 0.7821
  Coverage agreement  - mean κ       : 0.7634
  Decision agreement  - mean α       : 0.6912
  Decision agreement  - mean κ       : 0.6805
  Span matching       - mean F1      : 0.8103

Label                              cov α    dec α       F1
-----------------------------------------------------------------
Actors                             0.8234   0.7901   0.8450
Acts                               0.7654   0.6800   0.7900
...
```

**JSON report structure:**
```
report.json
├── meta/               # Run parameters and methodology notes
├── per_label/
│   └── <label>/
│       ├── span_counts_per_annotator
│       ├── coverage_agreement/
│       │   ├── krippendorff_alpha
│       │   └── cohen_kappa_pairs/
│       ├── decision_agreement/
│       │   ├── krippendorff_alpha
│       │   └── cohen_kappa_pairs/
│       └── span_matching/
│           ├── macro_f1
│           └── pairs/
│               └── annotator_X_vs_annotator_Y/
│                   ├── annotator_X_as_reference  (precision, recall, f1)
│                   └── annotator_Y_as_reference  (precision, recall, f1)
└── summary/            # Macro-averages across all labels + interpretation guide
```

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

None. Standard library only (`argparse`, `itertools`, `json`, `math`, `re`). Requires Python 3.10+.

---

## Considerations and Known Limitations

### 1. `--criterion exact` in View 2 is almost always too strict

Positive clusters are built by **merging all overlapping spans from all annotators**. This means a cluster's boundaries are the union of every annotator's span in that region. If annotators A and B annotate the same region with even slightly different boundaries, the cluster will be wider than either individual span. With `--criterion exact`, neither annotator's span will exactly match the cluster — both score `0` — producing misleading decision agreement scores. **`--criterion contained` is the recommended setting for View 2**, and `exact` is more meaningful for View 3 where individual spans are compared directly to each other.

### 2. Span matching uses greedy first-fit, not optimal matching

View 3 matches spans 1-to-1 with a greedy first-fit strategy (for each reference span, take the first unmatched system span that satisfies the criterion). This can produce suboptimal results when multiple spans could potentially match each other. An optimal bipartite matching (e.g. the Hungarian algorithm) would be more correct, especially in dense annotation regions with many overlapping spans.

### 3. The `contained` criterion in View 3 is bidirectional

`contained` in View 3 accepts a match if either span is a subset of the other: `[a,b] ⊆ [c,d]` **or** `[c,d] ⊆ [a,b]`. This means a very long span and a very short span will match, as long as one contains the other. Depending on your use case, you may want a stricter definition (e.g. requiring the system span to be contained within the reference, not the other way around).

### 4. Negative items are balanced 1:1 with positive clusters in View 2

For every positive cluster in a document, exactly one negative item (all 0s) is added. This keeps the ratio of annotated vs unannotated decisions at 1:1 regardless of label prevalence, avoiding inflation from many trivial 0-0 items in sparse labels while retaining a non-annotation baseline so the metric does not deflate when annotators frequently annotate different regions.

### 5. Cohen's κ is computed per pair, not multi-annotator

The tool reports κ for every unordered annotator pair. There is no single multi-annotator κ; the mean κ across pairs is reported as a summary statistic. Krippendorff's α is the appropriate multi-annotator generalisation and is also reported.

### 6. Partially assigned datasets

If annotators are assigned to different subsets of documents (which is common in large annotation projects), pairwise κ is only computed over documents where both annotators appear. This is handled correctly via `None` values in the reliability matrix, but it means pair-level scores may be based on very different numbers of documents — interpret with caution when assignment overlap is low.
