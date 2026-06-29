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

  span   The item universe is the set of unique span positions (start, end)
         seen across ALL annotators for a given label in a given document.
         An annotator gets 1 for a candidate if they marked it, 0 if they
         did not (meaning: this span boundary was discovered by a peer but
         they chose not to use it — a principled, non-phantom zero).
         Spans that no annotator marked never enter the universe, so there
         are no invented items.

         Here `criterion` is meaningful:
           exact           (start, end) must match exactly.
           fully_contained Two positions are merged into one candidate if
                           one fully contains the other; the wider span
                           becomes the canonical representative.

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
# Candidate span universe  (span granularity)
# ---------------------------------------------------------------------------

def build_span_universe(
    ann_map: dict[str, list[dict]],
    label: str,
    criterion: Criterion,
) -> list[tuple[int, int]]:
    """
    Collect all unique (start, end) positions for *label* seen by any
    annotator in this document.

    exact            Each distinct (start, end) pair is its own candidate.
    fully_contained  Positions where one contains the other are merged;
                     the wider span becomes the canonical representative.
                     Positions are processed largest-first so the wider
                     span is always seen first.
    """
    raw: list[tuple[int, int]] = []
    for anns in ann_map.values():
        for ann in anns:
            if ann["label"] == label:
                pos = (ann["start"], ann["end"])
                if pos not in raw:
                    raw.append(pos)

    if criterion == "exact" or not raw:
        return raw

    # fully_contained: merge positions where one contains the other
    # Sort by span length descending so wider spans are canonical
    raw_sorted = sorted(raw, key=lambda p: p[1] - p[0], reverse=True)
    canonical: list[tuple[int, int]] = []
    for pos in raw_sorted:
        absorbed = False
        for canon in canonical:
            # pos is fully inside an already-canonical span
            if canon[0] <= pos[0] and canon[1] >= pos[1]:
                absorbed = True
                break
            # canon is fully inside pos — replace canon with pos (wider)
            if pos[0] <= canon[0] and pos[1] >= canon[1]:
                canonical[canonical.index(canon)] = pos
                absorbed = True
                break
        if not absorbed:
            canonical.append(pos)
    return canonical


# ---------------------------------------------------------------------------
# Reliability matrix builders
# ---------------------------------------------------------------------------

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
    Item universe = union of (start, end) positions seen by any annotator,
    per document. Spans seen by nobody are never added (no phantom zeros).

    Rows = annotators, Cols = one entry per candidate span across all docs.
    1.0  annotator marked this span position with this label
    0.0  annotator was assigned this doc but did not mark this position
         (the boundary was discovered by a peer — a principled zero)
    None annotator not assigned to this document
    """
    rows: list[list[float | None]] = [[] for _ in annotators]

    for doc in documents:
        ann_map = {
            str(asgn["annotator"]): asgn["annotations"]
            for asgn in doc["assignments"]
        }
        universe = build_span_universe(ann_map, label, criterion)
        if not universe:
            continue

        assigned = set(ann_map.keys())

        for (cand_start, cand_end) in universe:
            for i, annotator in enumerate(annotators):
                if annotator not in assigned:
                    rows[i].append(None)
                else:
                    # Check whether annotator marked this exact canonical position
                    # (or a position that maps to it under fully_contained)
                    hit = False
                    for ann in ann_map[annotator]:
                        if ann["label"] != label:
                            continue
                        a_s, a_e = ann["start"], ann["end"]
                        if criterion == "exact":
                            if a_s == cand_start and a_e == cand_end:
                                hit = True
                                break
                        else:  # fully_contained
                            # The annotator's span and the canonical span
                            # overlap in containment
                            if (cand_start <= a_s and cand_end >= a_e) or \
                               (a_s <= cand_start and a_e >= cand_end):
                                hit = True
                                break
                    rows[i].append(1.0 if hit else 0.0)

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
                    "Each unique span position seen by at least one annotator is one item. "
                    "1=annotator marked it, 0=annotator was assigned the doc but didn't mark it, "
                    "None=doc not assigned. Spans seen by nobody are never added."
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
