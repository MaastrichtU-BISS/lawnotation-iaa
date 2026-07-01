"""
Inter-Annotator Agreement for Span Annotations
================================================
Reports THREE complementary views of agreement, computed per label:

1. COVERAGE AGREEMENT  (Krippendorff's alpha, Cohen's kappa — char/word)
   "How much text do annotators agree on?"
   The unit of analysis is a position in the text (a character or a word).
   Agreement on one long span contributes many agreeing items, while
   agreement on several short spans contributes fewer — i.e. it is
   LENGTH-WEIGHTED. Two annotators agreeing on one big 200-character span
   counts for more than agreeing on ten separate 5-character spans.

2. DECISION AGREEMENT  (Krippendorff's alpha, Cohen's kappa — clusters)
   "How consistently do annotators agree on annotation decisions?"
   The unit of analysis is an ANNOTATION CLUSTER: all spans (from any
   annotator) that touch or overlap each other are merged into one
   contiguous block, and that whole block counts as exactly ONE item,
   no matter how long it is. For each positive cluster one negative item
   is added, keeping a 1:1 balance between annotated and unannotated
   decisions and avoiding label-prevalence bias. An annotator scores
   1 on a cluster if any of their own spans overlap it, 0 otherwise. This
   removes length entirely from the calculation — one big agreed cluster
   and one small agreed cluster both count as a single agreeing item.

3. SPAN MATCHING  (Precision / Recall / F1)
   "If one annotator is treated as the reference, how accurately can
   another annotator reproduce their annotations?"
   The unit of analysis is a whole SPAN. One annotator is the reference,
   the other is evaluated against it; spans are matched 1-to-1, and
   unmatched spans count as misses/false positives. This is NOT
   chance-corrected, and is inherently pairwise (an annotator must be
   picked as reference), so it is reported per-pair, in both directions,
   plus a macro-average.

Matching criterion (applies to views 2 and 3)
------------------------------------------------
  exact            A match requires identical (start, end).
  contained  A match is accepted if one span fully contains the
                    other (treats minor boundary differences as agreement).

Granularity (only affects view 1)
------------------------------------
  char   Each character position is one item. Slower, finest-grained.
  word   Each whitespace-delimited word is one item. Faster, coarser.
         `criterion` has no effect here — a span either fully contains a
         token or it doesn't, regardless of "exact" vs "contained".

Missing values (None)
----------------------
An annotator is excluded (None) from an item only when they were not
assigned to the document containing it at all. This differs from 0, which
means "assigned, but did not annotate here."

Usage
-----
  python iaa_spans.py --input data.json [options]

  Options:
    --criterion   exact | contained  (default: exact)
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
Criterion   = Literal["exact", "contained"]
Granularity = Literal["char", "word"]


# ===========================================================================
# Shared agreement formulas — used by both view 1 (coverage) and
# view 2 (decision), since both reduce to a binary reliability matrix.
# ===========================================================================

def krippendorff_alpha(matrix: list[list[float | None]]) -> float | None:
    """
    Krippendorff's alpha with the nominal/binary distance metric
    (delta = 0 if values match, 1 otherwise), computed directly from a
    reliability matrix (rows = annotators, cols = items). Handles missing
    values (None) by excluding them from both the observed and expected
    disagreement calculations.
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
# VIEW 1 — Coverage agreement: char/word reliability matrix
# ===========================================================================

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


def build_coverage_matrix(
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


# ===========================================================================
# VIEW 2 — Decision agreement: cluster-based reliability matrix
# ===========================================================================

def build_clusters(
    ann_map: dict[str, list[dict]],
    label: str,
) -> list[tuple[int, int]]:
    """
    Merge every span (from ANY annotator) for *label* that touches or
    overlaps another into one contiguous positive cluster. Each cluster
    counts as exactly one item regardless of length. Unannotated regions
    are not represented — the matrix only contains items where at least
    one annotator made a decision, keeping the metric focused on
    annotation consistency rather than being distorted by label prevalence.
    """
    spans: list[tuple[int, int]] = []
    for anns in ann_map.values():
        for ann in anns:
            if ann["label"] == label:
                spans.append((ann["start"], ann["end"]))

    spans.sort()
    positive: list[tuple[int, int]] = []
    for s, e in spans:
        if positive and s <= positive[-1][1]:
            positive[-1] = (positive[-1][0], max(positive[-1][1], e))
        else:
            positive.append((s, e))

    return positive


def cluster_is_covered(
    cluster_start: int,
    cluster_end: int,
    annotations: list[dict],
    label: str,
    criterion: Criterion,
) -> bool:
    """True if any of the annotator's spans for *label* matches the cluster per criterion."""
    for ann in annotations:
        if ann["label"] != label:
            continue
        a_s, a_e = ann["start"], ann["end"]
        if criterion == "exact":
            if a_s == cluster_start and a_e == cluster_end:
                return True
        else:  # contained: any overlap is accepted
            if a_s < cluster_end and a_e > cluster_start:
                return True
    return False


def build_decision_matrix(
    documents: list[dict],
    label: str,
    annotators: list[str],
    criterion: Criterion,
) -> list[list[float | None]]:
    """
    Rows = annotators, Cols = positive clusters + balanced negative items.
    1.0  annotator has a span matching this cluster (per criterion)
    0.0  annotator was assigned to the doc but has no matching span
    None annotator not assigned to this document
    For each positive cluster one negative item (all 0s) is added, keeping
    the ratio of annotated-to-unannotated decisions at exactly 1:1 per
    document. This avoids both inflation (from too many trivial 0-0 items
    in sparse labels) and deflation (from no baseline of non-annotation).
    """
    rows: list[list[float | None]] = [[] for _ in annotators]

    for doc in documents:
        ann_map = {
            str(asgn["annotator"]): asgn["annotations"]
            for asgn in doc["assignments"]
        }
        assigned = set(ann_map.keys())

        positive = build_clusters(ann_map, label)

        for cs, ce in positive:
            for i, annotator in enumerate(annotators):
                if annotator not in assigned:
                    rows[i].append(None)
                else:
                    covered = cluster_is_covered(cs, ce, ann_map[annotator], label, criterion)
                    rows[i].append(1.0 if covered else 0.0)

        # Add one negative item per positive cluster (1:1 ratio).
        # This balances "decided to annotate" vs "decided not to annotate"
        # decisions without letting label sparsity inflate agreement.
        for _ in positive:
            for i, annotator in enumerate(annotators):
                rows[i].append(None if annotator not in assigned else 0.0)

    return rows


# ===========================================================================
# VIEW 3 — Span matching: Precision / Recall / F1
# ===========================================================================

def spans_match(a: tuple[int, int], b: tuple[int, int], criterion: Criterion) -> bool:
    """Whether two (start, end) spans count as the same annotation."""
    a_s, a_e = a
    b_s, b_e = b
    if criterion == "exact":
        return a_s == b_s and a_e == b_e
    # contained: one span must be a subset of the other
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
                "coverage_agreement": (
                    "Krippendorff's alpha / Cohen's kappa on a "
                    f"{granularity}-level reliability matrix. 1=token covered "
                    "by label, 0=not covered, None=doc not assigned. "
                    "LENGTH-WEIGHTED: agreement on long spans contributes more "
                    "items than agreement on short spans. Answers: how much "
                    "text do annotators agree on?"
                ),
                "decision_agreement": (
                    "Krippendorff's alpha / Cohen's kappa on a cluster-level "
                    "reliability matrix. Overlapping spans from any annotator "
                    "are merged into single positive clusters, each counting "
                    "as exactly one item regardless of length. One negative "
                    "(0-0) item is added per positive cluster, keeping a 1:1 "
                    "balance between annotated and unannotated decisions. "
                    "DECISION-WEIGHTED. Answers: how consistently do annotators "
                    "agree on annotation decisions, regardless of span length?"
                ),
                "span_matching": (
                    "Precision/Recall/F1 from matching whole spans between "
                    f"annotator pairs, using criterion='{criterion}'. Not "
                    "chance-corrected; reported per-pair in both directions "
                    "since precision/recall are asymmetric. Answers: if one "
                    "annotator is the reference, how accurately can another "
                    "reproduce their annotations?"
                ),
            },
        },
        "per_label": {},
        "summary": {},
    }

    cov_alpha_values:  list[float] = []
    cov_kappa_values:  list[float] = []
    dec_alpha_values:  list[float] = []
    dec_kappa_values:  list[float] = []
    f1_values:         list[float] = []

    for label in labels:
        counts = span_counts(documents, label, annotators)

        # --- View 1: coverage agreement ---
        cov_matrix = build_coverage_matrix(documents, label, annotators, granularity)
        cov_alpha  = krippendorff_alpha(cov_matrix)
        cov_kappas = cohen_kappa_all_pairs(cov_matrix, annotators)

        # --- View 2: decision agreement ---
        dec_matrix = build_decision_matrix(documents, label, annotators, criterion)
        dec_alpha  = krippendorff_alpha(dec_matrix)
        dec_kappas = cohen_kappa_all_pairs(dec_matrix, annotators)

        # --- View 3: span matching ---
        matching = span_matching_all_pairs(documents, label, annotators, criterion)
        label_f1s = [
            direction_result["f1"]
            for pair_result in matching.values()
            for direction_result in pair_result.values()
            if direction_result["f1"] is not None
        ]
        macro_f1 = round(sum(label_f1s) / len(label_f1s), 4) if label_f1s else None

        results["per_label"][label] = {
            "span_counts_per_annotator": counts,
            "coverage_agreement": {
                "matrix_items": len(cov_matrix[0]) if cov_matrix else 0,
                "krippendorff_alpha": round(cov_alpha, 4) if cov_alpha is not None else None,
                "cohen_kappa_pairs": {
                    k: round(v, 4) if v is not None else None
                    for k, v in cov_kappas.items()
                },
            },
            "decision_agreement": {
                "matrix_items": len(dec_matrix[0]) if dec_matrix else 0,
                "krippendorff_alpha": round(dec_alpha, 4) if dec_alpha is not None else None,
                "cohen_kappa_pairs": {
                    k: round(v, 4) if v is not None else None
                    for k, v in dec_kappas.items()
                },
            },
            "span_matching": {
                "macro_f1": macro_f1,
                "pairs": matching,
            },
        }

        if cov_alpha is not None:
            cov_alpha_values.append(cov_alpha)
        for v in cov_kappas.values():
            if v is not None:
                cov_kappa_values.append(v)
        if dec_alpha is not None:
            dec_alpha_values.append(dec_alpha)
        for v in dec_kappas.values():
            if v is not None:
                dec_kappa_values.append(v)
        if macro_f1 is not None:
            f1_values.append(macro_f1)

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    results["summary"] = {
        "coverage_agreement": {
            "mean_krippendorff_alpha":    safe_mean(cov_alpha_values),
            "mean_cohen_kappa_all_pairs": safe_mean(cov_kappa_values),
        },
        "decision_agreement": {
            "mean_krippendorff_alpha":    safe_mean(dec_alpha_values),
            "mean_cohen_kappa_all_pairs": safe_mean(dec_kappa_values),
        },
        "span_matching": {
            "mean_f1_all_labels": safe_mean(f1_values),
        },
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
    parser.add_argument("--criterion",   choices=["exact", "contained"], default="exact")
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

    s = report["summary"]
    print("=== SUMMARY ===")
    print(f"  Documents                          : {report['meta']['num_documents']}")
    print(f"  Annotators                         : {', '.join(report['meta']['annotators'])}")
    print(f"  Granularity (coverage view)        : {report['meta']['granularity']}")
    print(f"  Coverage agreement  - mean α        : {s['coverage_agreement']['mean_krippendorff_alpha']}")
    print(f"  Coverage agreement  - mean κ         : {s['coverage_agreement']['mean_cohen_kappa_all_pairs']}")
    print(f"  Decision agreement  - mean α        : {s['decision_agreement']['mean_krippendorff_alpha']}")
    print(f"  Decision agreement  - mean κ         : {s['decision_agreement']['mean_cohen_kappa_all_pairs']}")
    print(f"  Span matching       - mean F1       : {s['span_matching']['mean_f1_all_labels']}")
    print()

    col_w = 34
    print(f"{'Label':<{col_w}} {'cov α':>8} {'dec α':>8} {'F1':>8}")
    print("-" * 65)
    for label, v in report["per_label"].items():
        cov_a = v["coverage_agreement"]["krippendorff_alpha"]
        dec_a = v["decision_agreement"]["krippendorff_alpha"]
        f1    = v["span_matching"]["macro_f1"]
        cov_str = f"{cov_a:>8.4f}" if cov_a is not None else "     N/A"
        dec_str = f"{dec_a:>8.4f}" if dec_a is not None else "     N/A"
        f1_str  = f"{f1:>8.4f}" if f1 is not None else "     N/A"
        print(f"{label:<{col_w}}{cov_str}{dec_str}{f1_str}")


if __name__ == "__main__":
    main()