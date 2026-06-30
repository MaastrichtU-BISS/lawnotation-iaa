"""
Inter-Annotator Agreement for Span Annotations
================================================
Computes per-label Krippendorff's Alpha (all annotators) and Cohen's Kappa
(each pair).

Three granularity modes
-----------------------
  char   Each character position is one item. A token gets 1 if any span of
         the target label covers it, 0 otherwise. Fine-grained; slower on
         large files. `criterion` has no effect at this level (a character
         is a point — containment and exactness are identical).

  word   Each whitespace-delimited word is one item. Same 1/0 logic as char
         but coarser and faster. `criterion` is also inert here in practice
         because the coverage check (span.start <= tok.start AND
         span.end >= tok.end) resolves identically for both criteria.

  span   Items are built by SEGMENTATION, not by collecting raw span
         positions, and combine TWO kinds of segments:

         (1) POSITIVE segments — all span boundaries (start/end offsets)
             drawn by ANY annotator for the label, in the document, are
             used as cut points that partition the document into
             contiguous segments:

                 [ann1]text1[ann2]text2[ann3]...

             1=annotator's span covers the segment, 0=does not.

         (2) GAP segments (fictional "NOT-label" class) — for EACH
             annotator individually, their own gap intervals (the
             complement of their own merged spans for the label) are
             computed. These per-annotator gap-interval sets are then
             pooled to build segments the same way. To keep polarity
             consistent with the positive block (1=annotated, 0=not),
             a gap segment scores 1 if the annotator actually labeled
             it (i.e. it falls OUTSIDE their own gap interval) and 0 if
             it falls INSIDE their gap interval. A mutual gap — both
             annotators agree nothing is labeled there — therefore
             correctly reads as 0-0, not 1-1. This avoids collapsing
             every unannotated stretch into one giant shared region —
             disagreement on exactly WHERE the gaps are is captured at
             the same resolution as the positive label, instead of being
             flattened into a single coarse "nobody annotated here" item.

         Both kinds of segments become columns in the same matrix.

         `criterion` currently only affects how POSITIVE boundaries are
         collected:
           exact            every distinct start/end offset is its own cut
                            point.
           fully_contained  boundaries strictly inside an already-canonical
                            (wider) span from another annotator are
                            dropped. NOTE: this can destroy small islands
                            of perfect agreement nested inside a wider
                            span from a different annotator — use with
                            caution, `exact` is the safer default.

Missing values (None)
---------------------
An annotator gets None (excluded from all calculations) when they were not
assigned to a document at all. This differs from 0, which means "assigned
but did not label this item."

Usage
-----
  python iaa_spans.py --input data.json [options]

  Options:
    --criterion   exact | fully_contained  (default: exact)
    --granularity char | word | span       (default: word)
    --output      path to JSON report      (default: iaa_report.json)
"""

import argparse
import itertools
import json
import math
import re
from typing import Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Criterion   = Literal["exact", "fully_contained"]
Granularity = Literal["char", "word", "span"]


# ---------------------------------------------------------------------------
# Tokenisation  (char / word granularity)
# ---------------------------------------------------------------------------

def tokenize(text: str, granularity: Granularity) -> list[tuple[int, int]]:
    """Return (start, end) character offsets for each token."""
    if granularity == "char":
        return [(i, i + 1) for i in range(len(text))]
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


# ---------------------------------------------------------------------------
# Token coverage  (char / word granularity)
# ---------------------------------------------------------------------------

def token_is_covered(
    tok_start: int,
    tok_end: int,
    annotations: list[dict],
    label: str,
) -> bool:
    """True if any annotation of *label* fully contains the token."""
    for ann in annotations:
        if ann["label"] == label:
            if ann["start"] <= tok_start and ann["end"] >= tok_end:
                return True
    return False


# ---------------------------------------------------------------------------
# Boundary segmentation  (span granularity)
# ---------------------------------------------------------------------------

def build_segments(
    doc_length: int,
    ann_map: dict[str, list[dict]],
    label: str,
    criterion: Criterion,
) -> list[tuple[int, int]]:
    """
    Collect every start/end boundary drawn by any annotator for *label* in
    this document, use them as cut points, and partition [0, doc_length)
    into contiguous, non-overlapping segments.

    exact            Every distinct boundary offset is its own cut point.
    fully_contained  Boundaries strictly inside an already-canonical
                     (wider) span from another annotator are dropped, so
                     near-identical spans collapse into one segment.

    Segments with zero length (caused by duplicate boundaries, e.g. two
    annotators using the exact same start/end) are skipped.
    """
    spans: list[tuple[int, int]] = []
    for anns in ann_map.values():
        for ann in anns:
            if ann["label"] == label:
                spans.append((ann["start"], ann["end"]))

    if not spans:
        # Nobody annotated this label in this doc — whole document is one
        # shared 0 segment for every assigned annotator.
        return [(0, doc_length)] if doc_length > 0 else []

    if criterion == "fully_contained":
        # Drop boundaries that fall strictly inside a wider span from
        # another annotator, so minor boundary differences collapse.
        spans_sorted = sorted(spans, key=lambda p: p[1] - p[0], reverse=True)
        canonical: list[tuple[int, int]] = []
        for s, e in spans_sorted:
            absorbed = any(cs <= s and ce >= e for cs, ce in canonical)
            if not absorbed:
                canonical.append((s, e))
        spans = canonical

    boundaries = {0, doc_length}
    for s, e in spans:
        boundaries.add(max(0, min(s, doc_length)))
        boundaries.add(max(0, min(e, doc_length)))

    sorted_bounds = sorted(boundaries)
    segments = [
        (sorted_bounds[i], sorted_bounds[i + 1])
        for i in range(len(sorted_bounds) - 1)
        if sorted_bounds[i] < sorted_bounds[i + 1]
    ]
    return segments


# ---------------------------------------------------------------------------
# Per-annotator gap intervals  (the "NOT-label" fictional class)
# ---------------------------------------------------------------------------

def merged_covered_intervals(
    annotations: list[dict],
    label: str,
) -> list[tuple[int, int]]:
    """Merge an annotator's own (possibly overlapping) spans for *label*."""
    ivs = sorted(
        (ann["start"], ann["end"]) for ann in annotations if ann["label"] == label
    )
    merged: list[tuple[int, int]] = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def complement_intervals(
    intervals: list[tuple[int, int]],
    doc_length: int,
) -> list[tuple[int, int]]:
    """Return the gaps (complement) of a sorted, merged interval list."""
    comp: list[tuple[int, int]] = []
    prev = 0
    for s, e in intervals:
        if s > prev:
            comp.append((prev, s))
        prev = max(prev, e)
    if prev < doc_length:
        comp.append((prev, doc_length))
    return comp


def build_gap_segments(
    doc_length: int,
    ann_map: dict[str, list[dict]],
    label: str,
) -> tuple[list[tuple[int, int]], dict[str, list[tuple[int, int]]]]:
    """
    For each annotator, compute their own gap intervals (the complement of
    their merged spans for *label*) — this is the "NOT-label" fictional
    class, one distinct interval set PER annotator, not a single shared
    leftover region.

    All annotators' gap-interval boundaries are then pooled to build
    segments, exactly like build_segments does for the positive label.
    This lets disagreement on WHERE the gaps are (not just whether a gap
    exists) be captured at the same resolution as the positive label.

    Returns (segments, gaps_by_annotator).
    """
    gaps_by_annotator: dict[str, list[tuple[int, int]]] = {}
    for annotator, anns in ann_map.items():
        covered = merged_covered_intervals(anns, label)
        gaps_by_annotator[annotator] = complement_intervals(covered, doc_length)

    boundaries = {0, doc_length}
    for ivs in gaps_by_annotator.values():
        for s, e in ivs:
            boundaries.add(max(0, min(s, doc_length)))
            boundaries.add(max(0, min(e, doc_length)))

    sorted_bounds = sorted(boundaries)
    segments = [
        (sorted_bounds[i], sorted_bounds[i + 1])
        for i in range(len(sorted_bounds) - 1)
        if sorted_bounds[i] < sorted_bounds[i + 1]
    ]
    return segments, gaps_by_annotator


def gap_segment_covered(
    seg_start: int,
    seg_end: int,
    gap_intervals: list[tuple[int, int]],
) -> bool:
    """True if one of this annotator's own gap intervals fully covers the segment."""
    return any(gs <= seg_start and ge >= seg_end for gs, ge in gap_intervals)


def build_token_matrix(
    documents: list[dict],
    label: str,
    annotators: list[str],
    granularity: Granularity,   # "char" or "word"
) -> list[list[float | None]]:
    """
    Rows = annotators, Cols = one entry per token across all documents.
    1.0  annotator's span covers this token for this label
    0.0  annotator worked on doc but did not cover the token
    None annotator not assigned to this document
    """
    rows: list[list[float | None]] = [[] for _ in annotators]

    for doc in documents:
        text   = doc["full_text"]
        tokens = tokenize(text, granularity)
        ann_map = {
            str(asgn["annotator"]): asgn["annotations"]
            for asgn in doc["assignments"]
        }
        for tok_start, tok_end in tokens:
            for i, annotator in enumerate(annotators):
                if annotator not in ann_map:
                    rows[i].append(None)
                else:
                    covered = token_is_covered(
                        tok_start, tok_end, ann_map[annotator], label
                    )
                    rows[i].append(1.0 if covered else 0.0)

    return rows


def build_span_matrix(
    documents: list[dict],
    label: str,
    annotators: list[str],
    criterion: Criterion,
) -> list[list[float | None]]:
    """
    Item universe = two kinds of segments, concatenated as columns:

    1. POSITIVE segments — the document cut at every boundary any
       annotator drew for *label* (see build_segments). 1=annotator's
       span covers the segment, 0=does not.

    2. GAP segments — for each annotator, their own gap intervals (the
       complement of their merged spans for *label*) form a distinct,
       per-annotator "NOT-label" interval set. These interval sets are
       pooled across annotators to build segments the same way, and
       1=this annotator's OWN gap interval covers the segment, 0=does
       not (meaning some part of an actual span intrudes there).

       This avoids collapsing every unannotated stretch into one giant
       shared region — disagreement on exactly WHERE the gaps are is
       captured at the same resolution as the positive label, and
       perfect-agreement islands inside a longer span (e.g. a small
       span everyone nests inside a bigger one) are not destroyed by
       indiscriminate merging.

    Rows = annotators, Cols = positive segments ++ gap segments, all docs.
    None = annotator not assigned to this document.
    """
    rows: list[list[float | None]] = [[] for _ in annotators]

    for doc in documents:
        text = doc["full_text"]
        ann_map = {
            str(asgn["annotator"]): asgn["annotations"]
            for asgn in doc["assignments"]
        }
        assigned = set(ann_map.keys())

        # --- positive segments ---
        pos_segments = build_segments(len(text), ann_map, label, criterion)
        for seg_start, seg_end in pos_segments:
            for i, annotator in enumerate(annotators):
                if annotator not in assigned:
                    rows[i].append(None)
                else:
                    covered = token_is_covered(
                        seg_start, seg_end, ann_map[annotator], label
                    )
                    rows[i].append(1.0 if covered else 0.0)

        # --- gap segments (fictional "NOT-label" class) ---
        # NOTE on polarity: gap_segment_covered() returning True means this
        # annotator's gap interval covers the segment, i.e. they did NOT
        # annotate there. To keep 1 meaning "annotated" and 0 meaning "gap"
        # consistently across the whole matrix (so mutual gaps correctly
        # read as 0-0, not 1-1), we INVERT the boolean here.
        gap_segments, gaps_by_annotator = build_gap_segments(
            len(text), ann_map, label
        )
        for seg_start, seg_end in gap_segments:
            for i, annotator in enumerate(annotators):
                if annotator not in assigned:
                    rows[i].append(None)
                else:
                    is_gap = gap_segment_covered(
                        seg_start, seg_end, gaps_by_annotator[annotator]
                    )
                    rows[i].append(0.0 if is_gap else 1.0)

    return rows


def build_reliability_matrix(
    documents: list[dict],
    label: str,
    annotators: list[str],
    granularity: Granularity,
    criterion: Criterion,
) -> list[list[float | None]]:
    if granularity == "span":
        return build_span_matrix(documents, label, annotators, criterion)
    return build_token_matrix(documents, label, annotators, granularity)


# ---------------------------------------------------------------------------
# Krippendorff's Alpha  (nominal, missing-data-aware)
# ---------------------------------------------------------------------------

def krippendorff_alpha(matrix: list[list[float | None]]) -> float | None:
    if not matrix or not matrix[0]:
        return None

    n_annotators = len(matrix)
    n_items = len(matrix[0])

    do_num = do_den = 0.0
    all_vals: list[float] = []

    for col in range(n_items):
        col_vals = [matrix[r][col] for r in range(n_annotators)
                    if matrix[r][col] is not None]
        all_vals.extend(col_vals)
        m_u = len(col_vals)
        if m_u < 2:
            continue
        for i in range(m_u):
            for j in range(i + 1, m_u):
                do_num += 0.0 if col_vals[i] == col_vals[j] else 1.0
                do_den += 1.0

    if do_den == 0:
        return None

    Do = do_num / do_den
    n  = len(all_vals)
    if n < 2:
        return None

    c1 = sum(all_vals)
    c0 = n - c1
    De = 2.0 * c1 * c0 / (n * (n - 1))

    if De == 0.0:
        return 1.0 if Do == 0.0 else None

    return 1.0 - Do / De


# ---------------------------------------------------------------------------
# Cohen's Kappa  (binary, pairwise)
# ---------------------------------------------------------------------------

def cohen_kappa_pair(
    vals_a: list[float | None],
    vals_b: list[float | None],
) -> float | None:
    paired = [(a, b) for a, b in zip(vals_a, vals_b)
              if a is not None and b is not None]
    if len(paired) < 2:
        return None

    n   = len(paired)
    p_o = sum(1 for a, b in paired if a == b) / n
    p_a = sum(a for a, _ in paired) / n
    p_b = sum(b for _, b in paired) / n
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)

    if math.isclose(p_e, 1.0):
        return 1.0 if math.isclose(p_o, 1.0) else None

    return (p_o - p_e) / (1.0 - p_e)


def cohen_kappa_all_pairs(
    matrix: list[list[float | None]],
    annotators: list[str],
) -> dict[str, float | None]:
    return {
        f"annotator_{annotators[i]}_vs_annotator_{annotators[j]}":
            cohen_kappa_pair(matrix[i], matrix[j])
        for i, j in itertools.combinations(range(len(annotators)), 2)
    }


# ---------------------------------------------------------------------------
# Span count helper
# ---------------------------------------------------------------------------

def span_counts(
    documents: list[dict],
    label: str,
    annotators: list[str],
) -> dict[str, int]:
    counts = {f"annotator_{a}": 0 for a in annotators}
    for doc in documents:
        for asgn in doc["assignments"]:
            key = f"annotator_{asgn['annotator']}"
            if key in counts:
                counts[key] += sum(
                    1 for ann in asgn["annotations"] if ann["label"] == label
                )
    return counts


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple[list[str], list[str], list[dict]]:
    with open(path) as f:
        raw = json.load(f)
    labels = [lbl["name"] for lbl in raw["labelset"]["labels"]]
    annotators = sorted({
        str(asgn["annotator"])
        for doc in raw["documents"]
        for asgn in doc["assignments"]
    })
    return labels, annotators, raw["documents"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_iaa(
    input_path: str,
    criterion: Criterion,
    granularity: Granularity,
) -> dict:
    labels, annotators, documents = load_data(input_path)

    criterion_note = (
        "criterion is not applicable at char/word granularity "
        "(token containment is identical for both criteria)"
        if granularity in ("char", "word") else
        f"criterion='{criterion}' controls how span positions are matched "
        "when building the candidate universe"
    )

    results: dict = {
        "meta": {
            "input_file":    input_path,
            "criterion":     criterion,
            "granularity":   granularity,
            "annotators":    [f"annotator_{a}" for a in annotators],
            "num_documents": len(documents),
            "matrix_note": {
                "char/word": (
                    "Each token is one item. "
                    "1=label covers token, 0=token not covered, None=doc not assigned."
                ),
                "span": (
                    "Combines two segment types: (1) positive segments from "
                    "cutting the document at every label boundary any annotator "
                    "drew, 1=span covers segment; (2) gap segments built from "
                    "each annotator's OWN complement intervals (per-annotator, "
                    "not one shared blob), 1=annotator's own gap covers segment. "
                    "None=doc not assigned."
                ),
            }[granularity if granularity in ("span",) else "char/word"],
            "criterion_note": criterion_note,
        },
        "per_label": {},
        "summary": {},
    }

    alpha_values: list[float] = []
    kappa_values: list[float] = []

    for label in labels:
        matrix = build_reliability_matrix(
            documents, label, annotators, granularity, criterion
        )
        n_items = len(matrix[0]) if matrix else 0

        alpha  = krippendorff_alpha(matrix)
        kappas = cohen_kappa_all_pairs(matrix, annotators)
        counts = span_counts(documents, label, annotators)

        results["per_label"][label] = {
            "span_counts_per_annotator": counts,
            "matrix_items": n_items,
            "krippendorff_alpha": round(alpha, 4) if alpha is not None else None,
            "cohen_kappa_pairs": {
                k: round(v, 4) if v is not None else None
                for k, v in kappas.items()
            },
        }

        if alpha is not None:
            alpha_values.append(alpha)
        for v in kappas.values():
            if v is not None:
                kappa_values.append(v)

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    results["summary"] = {
        "mean_krippendorff_alpha":    safe_mean(alpha_values),
        "mean_cohen_kappa_all_pairs": safe_mean(kappa_values),
        "interpretation_guide": {
            "< 0.20":      "Slight agreement",
            "0.21-0.40":   "Fair agreement",
            "0.41-0.60":   "Moderate agreement",
            "0.61-0.80":   "Substantial agreement",
            "> 0.80":      "Almost perfect agreement",
        },
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inter-annotator agreement for span annotations."
    )
    parser.add_argument("--input",       required=True)
    parser.add_argument("--criterion",   choices=["exact", "fully_contained"], default="exact")
    parser.add_argument("--granularity", choices=["char", "word", "span"],     default="word")
    parser.add_argument("--output",      default="iaa_report.json")
    args = parser.parse_args()

    print(f"Input        : {args.input}")
    print(f"Criterion    : {args.criterion}"
          + (" (n/a at char/word granularity)" if args.granularity != "span" else ""))
    print(f"Granularity  : {args.granularity}")
    print(f"Output       : {args.output}")
    print()

    report = compute_iaa(args.input, args.criterion, args.granularity)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to: {args.output}\n")
    print("=== SUMMARY ===")
    print(f"  Documents           : {report['meta']['num_documents']}")
    print(f"  Annotators          : {', '.join(report['meta']['annotators'])}")
    print(f"  Granularity         : {report['meta']['granularity']}")
    print(f"  Mean Krippendorff α : {report['summary']['mean_krippendorff_alpha']}")
    print(f"  Mean Cohen κ        : {report['summary']['mean_cohen_kappa_all_pairs']}")
    print()

    col_w = 42
    print(f"{'Label':<{col_w}} {'α':>8}  κ pairs")
    print("-" * 100)
    for label, v in report["per_label"].items():
        alpha_str = f"{v['krippendorff_alpha']:>8.4f}" if v["krippendorff_alpha"] is not None else "     N/A"
        kappa_parts = [
            f"{pk.replace('annotator_','').replace('_vs_',' vs ')}: "
            f"{pv:.4f}" if pv is not None else "N/A"
            for pk, pv in v["cohen_kappa_pairs"].items()
        ]
        print(f"{label:<{col_w}}{alpha_str}  {'  |  '.join(kappa_parts)}")


if __name__ == "__main__":
    main()