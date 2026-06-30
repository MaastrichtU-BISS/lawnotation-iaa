"""
Inter-Annotator Agreement for Span Annotations
================================================
Reports TWO complementary views of agreement, computed per label:

1. CHANCE-CORRECTED AGREEMENT (Krippendorff's alpha, Cohen's kappa)
   Measured at char or word granularity. The unit of analysis is a
   position in the text (a character or a word). This means agreement on
   one long span contributes many agreeing items, while agreement on
   several short spans contributes fewer — i.e. it is LENGTH-WEIGHTED.
   Two annotators agreeing on one big 200-character span counts for more
   than agreeing on ten separate 5-character spans. This is a deliberate,
   well-understood property: it reflects how much of the *text* annotators
   actually agree on, not how many *decisions* they made.

2. SPAN-MATCHING AGREEMENT (Precision / Recall / F1)
   The unit of analysis is an annotated SPAN itself, not a text position.
   One annotator is treated as the reference and the other as the system
   being evaluated; spans are matched 1-to-1 between them, and unmatched
   spans count as misses/false positives. A single long matched span and
   a single short matched span both count as exactly ONE match — so this
   view answers "how often do annotators agree a given annotation exists
   at all", independent of how long any individual span is. Unlike
   Krippendorff/Cohen, this is NOT chance-corrected, and it is inherently
   pairwise (an annotator must be picked as reference), so for 3+
   annotators it is reported per-pair, plus a macro-average.

Together these two views answer different questions: (1) how much of the
document's text do annotators agree on, weighted by coverage; (2) how often
do annotators agree an annotation exists, weighted by count not length.

Matching criterion (applies to both views)
-------------------------------------------
  exact            A span/segment match requires identical (start, end).
  fully_contained  A match is accepted if one span fully contains the
                    other (handles minor boundary differences as agreement).

Granularity (only affects the chance-corrected view)
------------------------------------------------------
  char   Each character position is one item. Slower, finest-grained.
  word   Each whitespace-delimited word is one item. Faster, coarser.
         `criterion` has no effect on char/word coverage checks — a span
         either fully contains a token or it doesn't, regardless of
         labelling it "exact" vs "fully_contained".

Missing values (None)
----------------------
An annotator is excluded (None) from an item only when they were not
assigned to the document containing it at all. This differs from 0, which
means "assigned, but did not annotate here."

Usage
-----
  python iaa_spans.py --input data.json [options]

  Options:
    --criterion   exact | fully_contained  (default: exact)
    --granularity char | word              (default: word)
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
Granularity = Literal["char", "word"]
Span        = tuple[int, int, str]  # (start, end, label)


# ===========================================================================
# PART 1 — Chance-corrected agreement (Krippendorff's alpha, Cohen's kappa)
#          via a char/word-level reliability matrix
# ===========================================================================

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(text: str, granularity: Granularity) -> list[tuple[int, int]]:
    """Return (start, end) character offsets for each token."""
    if granularity == "char":
        return [(i, i + 1) for i in range(len(text))]
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


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


def build_token_matrix(
    documents: list[dict],
    label: str,
    annotators: list[str],
    granularity: Granularity,
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


# ---------------------------------------------------------------------------
# Krippendorff's Alpha  (nominal, missing-data-aware, hand-rolled)
# ---------------------------------------------------------------------------

def krippendorff_alpha(matrix: list[list[float | None]]) -> float | None:
    """
    Krippendorff's alpha with the nominal/binary distance metric
    (delta = 0 if values match, 1 otherwise), computed directly from the
    reliability matrix. Handles missing values (None) by excluding them
    from both the observed and expected disagreement calculations.
    """
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
# Cohen's Kappa  (binary, pairwise, hand-rolled)
# ---------------------------------------------------------------------------

def cohen_kappa_pair(
    vals_a: list[float | None],
    vals_b: list[float | None],
) -> float | None:
    """
    Cohen's kappa between two annotators over a set of binary items. Items
    where either annotator has None (not assigned to that document) are
    excluded.
    """
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


# ===========================================================================
# PART 2 — Span-matching agreement (Precision / Recall / F1)
# ===========================================================================

def spans_match(a: tuple[int, int], b: tuple[int, int], criterion: Criterion) -> bool:
    """Whether two (start, end) spans count as the same annotation."""
    a_s, a_e = a
    b_s, b_e = b
    if criterion == "exact":
        return a_s == b_s and a_e == b_e
    # fully_contained: one span must be a subset of the other
    return (a_s >= b_s and a_e <= b_e) or (b_s >= a_s and b_e <= a_e)


def match_span_sets(
    spans_ref: list[tuple[int, int]],
    spans_sys: list[tuple[int, int]],
    criterion: Criterion,
) -> tuple[int, int, int]:
    """
    Greedily match spans_sys against spans_ref (1-to-1, first-fit).

    Returns (true_positives, ref_unmatched, sys_unmatched):
      true_positives  number of matched pairs
      ref_unmatched   reference spans with no match  (-> recall misses)
      sys_unmatched   system spans with no match     (-> precision misses)
    """
    matched_ref: set[int] = set()
    matched_sys: set[int] = set()

    for i, rs in enumerate(spans_ref):
        for j, ss in enumerate(spans_sys):
            if j in matched_sys:
                continue
            if spans_match(rs, ss, criterion):
                matched_ref.add(i)
                matched_sys.add(j)
                break

    tp = len(matched_ref)
    ref_unmatched = len(spans_ref) - tp
    sys_unmatched = len(spans_sys) - tp
    return tp, ref_unmatched, sys_unmatched


def precision_recall_f1(
    documents: list[dict],
    label: str,
    annotator_ref: str,
    annotator_sys: str,
    criterion: Criterion,
) -> dict[str, float | None]:
    """
    Compute precision/recall/F1 for annotator_sys against annotator_ref,
    for *label*, pooling matches across all documents both were assigned.
    Each matched span counts as one unit regardless of its length.
    """
    total_tp = total_ref_unmatched = total_sys_unmatched = 0
    n_docs_compared = 0

    for doc in documents:
        ann_map = {
            str(asgn["annotator"]): asgn["annotations"]
            for asgn in doc["assignments"]
        }
        if annotator_ref not in ann_map or annotator_sys not in ann_map:
            continue
        n_docs_compared += 1

        spans_ref = [(a["start"], a["end"]) for a in ann_map[annotator_ref] if a["label"] == label]
        spans_sys = [(a["start"], a["end"]) for a in ann_map[annotator_sys] if a["label"] == label]

        tp, ref_un, sys_un = match_span_sets(spans_ref, spans_sys, criterion)
        total_tp += tp
        total_ref_unmatched += ref_un
        total_sys_unmatched += sys_un

    n_ref = total_tp + total_ref_unmatched
    n_sys = total_tp + total_sys_unmatched

    precision = total_tp / n_sys if n_sys > 0 else None
    recall    = total_tp / n_ref if n_ref > 0 else None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    elif precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = None

    return {
        "true_positives":    total_tp,
        "ref_span_count":    n_ref,
        "sys_span_count":    n_sys,
        "precision":         round(precision, 4) if precision is not None else None,
        "recall":            round(recall, 4) if recall is not None else None,
        "f1":                round(f1, 4) if f1 is not None else None,
        "documents_compared": n_docs_compared,
    }


def span_matching_all_pairs(
    documents: list[dict],
    label: str,
    annotators: list[str],
    criterion: Criterion,
) -> dict[str, dict]:
    """
    For every unordered annotator pair, compute precision/recall/F1 in
    BOTH directions (each annotator taking a turn as reference), since
    precision/recall are inherently asymmetric.
    """
    results: dict[str, dict] = {}
    for i, j in itertools.combinations(range(len(annotators)), 2):
        a, b = annotators[i], annotators[j]
        key = f"annotator_{a}_vs_annotator_{b}"
        results[key] = {
            f"annotator_{a}_as_reference": precision_recall_f1(documents, label, a, b, criterion),
            f"annotator_{b}_as_reference": precision_recall_f1(documents, label, b, a, criterion),
        }
    return results


# ===========================================================================
# Shared helpers
# ===========================================================================

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


# ===========================================================================
# Main computation
# ===========================================================================

def compute_iaa(
    input_path: str,
    criterion: Criterion,
    granularity: Granularity,
) -> dict:
    labels, annotators, documents = load_data(input_path)

    results: dict = {
        "meta": {
            "input_file":    input_path,
            "criterion":     criterion,
            "granularity":   granularity,
            "annotators":    [f"annotator_{a}" for a in annotators],
            "num_documents": len(documents),
            "notes": {
                "chance_corrected": (
                    "Krippendorff's alpha and Cohen's kappa, computed on a "
                    f"{granularity}-level reliability matrix. 1=token covered "
                    "by label, 0=not covered, None=doc not assigned. "
                    "LENGTH-WEIGHTED: agreement on long spans contributes more "
                    "items than agreement on short spans."
                ),
                "span_matching": (
                    "Precision/Recall/F1 from matching whole spans between "
                    "annotator pairs, using criterion="
                    f"'{criterion}'. DECISION-WEIGHTED: each matched span "
                    "counts as one unit regardless of its length. Not "
                    "chance-corrected; reported per-pair since precision/"
                    "recall are asymmetric (one annotator is the reference)."
                ),
            },
        },
        "per_label": {},
        "summary": {},
    }

    alpha_values: list[float] = []
    kappa_values: list[float] = []
    f1_values: list[float] = []

    for label in labels:
        matrix = build_token_matrix(documents, label, annotators, granularity)
        n_items = len(matrix[0]) if matrix else 0

        alpha  = krippendorff_alpha(matrix)
        kappas = cohen_kappa_all_pairs(matrix, annotators)
        counts = span_counts(documents, label, annotators)
        matching = span_matching_all_pairs(documents, label, annotators, criterion)

        # Collect F1s for the label-level / overall macro average
        label_f1s = [
            direction_result["f1"]
            for pair_result in matching.values()
            for direction_result in pair_result.values()
            if direction_result["f1"] is not None
        ]
        macro_f1 = round(sum(label_f1s) / len(label_f1s), 4) if label_f1s else None

        results["per_label"][label] = {
            "span_counts_per_annotator": counts,
            "chance_corrected": {
                "matrix_items": n_items,
                "krippendorff_alpha": round(alpha, 4) if alpha is not None else None,
                "cohen_kappa_pairs": {
                    k: round(v, 4) if v is not None else None
                    for k, v in kappas.items()
                },
            },
            "span_matching": {
                "macro_f1": macro_f1,
                "pairs": matching,
            },
        }

        if alpha is not None:
            alpha_values.append(alpha)
        for v in kappas.values():
            if v is not None:
                kappa_values.append(v)
        if macro_f1 is not None:
            f1_values.append(macro_f1)

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    results["summary"] = {
        "mean_krippendorff_alpha":    safe_mean(alpha_values),
        "mean_cohen_kappa_all_pairs": safe_mean(kappa_values),
        "mean_f1_all_labels":         safe_mean(f1_values),
        "interpretation_guide_alpha_kappa": {
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
    parser.add_argument("--granularity", choices=["char", "word"],             default="word")
    parser.add_argument("--output",      default="iaa_report.json")
    args = parser.parse_args()

    print(f"Input        : {args.input}")
    print(f"Criterion    : {args.criterion}")
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
    print(f"  Mean F1 (macro)     : {report['summary']['mean_f1_all_labels']}")
    print()

    col_w = 38
    print(f"{'Label':<{col_w}} {'α':>8}  {'macro F1':>9}   κ pairs")
    print("-" * 110)
    for label, v in report["per_label"].items():
        cc = v["chance_corrected"]
        sm = v["span_matching"]
        alpha_str = f"{cc['krippendorff_alpha']:>8.4f}" if cc["krippendorff_alpha"] is not None else "     N/A"
        f1_str    = f"{sm['macro_f1']:>9.4f}" if sm["macro_f1"] is not None else "      N/A"
        kappa_parts = [
            f"{pk.replace('annotator_','').replace('_vs_',' vs ')}: "
            f"{pv:.4f}" if pv is not None else "N/A"
            for pk, pv in cc["cohen_kappa_pairs"].items()
        ]
        print(f"{label:<{col_w}}{alpha_str}  {f1_str}   {'  |  '.join(kappa_parts)}")


if __name__ == "__main__":
    main()