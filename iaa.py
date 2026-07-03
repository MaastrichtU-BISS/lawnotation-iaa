import argparse
import itertools
import json
import math
import re
from typing import Literal

Criterion   = Literal["exact", "contained"]
Granularity = Literal["char", "word"]

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



def tokenize(text: str, granularity: Granularity) -> list[tuple[int, int]]:
    if granularity == "char":
        return [(i, i + 1) for i in range(len(text))]
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def token_is_covered(
    tok_start: int,
    tok_end: int,
    annotations: list[dict],
    label: str,
) -> bool:
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


def spans_match(a: tuple[int, int], b: tuple[int, int], criterion: Criterion) -> bool:
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
    Maximum bipartite matching between reference and system spans.
    Finds the assignment that maximises true positives globally,
    avoiding the underestimation that greedy first-fit produces when
    spans overlap or nest (common with criterion='contained').
    Uses augmenting-path DFS — O(V * E), no external dependencies.

    Returns (true_positives, ref_unmatched, sys_unmatched).
    """
    adj = [
        [j for j, ss in enumerate(spans_sys) if spans_match(rs, ss, criterion)]
        for rs in spans_ref
    ]
    match_sys: list[int] = [-1] * len(spans_sys)

    def augment(i: int, visited: set[int]) -> bool:
        for j in adj[i]:
            if j in visited:
                continue
            visited.add(j)
            if match_sys[j] == -1 or augment(match_sys[j], visited):
                match_sys[j] = i
                return True
        return False

    tp = sum(1 for i in range(len(spans_ref)) if augment(i, set()))
    return tp, len(spans_ref) - tp, len(spans_sys) - tp


def precision_recall_f1(
    documents: list[dict],
    label: str,
    annotator_ref: str,
    annotator_sys: str,
    criterion: Criterion,
) -> dict[str, float | None]:
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
    results: dict[str, dict] = {}
    for i, j in itertools.combinations(range(len(annotators)), 2):
        a, b = annotators[i], annotators[j]
        key = f"annotator_{a}_vs_annotator_{b}"
        results[key] = {
            f"annotator_{a}_as_reference": precision_recall_f1(documents, label, a, b, criterion),
            f"annotator_{b}_as_reference": precision_recall_f1(documents, label, b, a, criterion),
        }
    return results

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
                "span_matching": (
                    "Precision/Recall/F1 from matching whole spans between "
                    f"annotator pairs, using criterion='{criterion}'. Not "
                    "chance-corrected; reported per-pair in both directions "
                    "since precision/recall are asymmetric. Answers: if one "
                    "annotator is the reference, how accurately can another "
                    "reproduce their annotations?"
                ),
                "coverage_agreement": (
                    "Krippendorff's alpha / Cohen's kappa on a "
                    f"{granularity}-level reliability matrix. 1=token covered "
                    "by label, 0=not covered, None=doc not assigned. "
                    "LENGTH-WEIGHTED: agreement on long spans contributes more "
                    "items than agreement on short spans. Answers: how much "
                    "text do annotators agree on?"
                ),
            },
        },
        "per_label": {},
        "summary": {},
    }

    cov_alpha_values:  list[float] = []
    cov_kappa_values:  list[float] = []
    f1_values:         list[float] = []

    for label in labels:
        counts = span_counts(documents, label, annotators)

        # --- View 1: span matching ---
        matching = span_matching_all_pairs(documents, label, annotators, criterion)
        label_f1s = [
            direction_result["f1"]
            for pair_result in matching.values()
            for direction_result in pair_result.values()
            if direction_result["f1"] is not None
        ]
        macro_f1 = round(sum(label_f1s) / len(label_f1s), 4) if label_f1s else None

        # --- View 2: coverage agreement ---
        cov_matrix = build_coverage_matrix(documents, label, annotators, granularity)
        cov_alpha  = krippendorff_alpha(cov_matrix)
        cov_kappas = cohen_kappa_all_pairs(cov_matrix, annotators)

        results["per_label"][label] = {
            "span_counts_per_annotator": counts,
            "span_matching": {
                "macro_f1": macro_f1,
                "pairs": matching,
            },
            "coverage_agreement": {
                "matrix_items": len(cov_matrix[0]) if cov_matrix else 0,
                "krippendorff_alpha": round(cov_alpha, 4) if cov_alpha is not None else None,
                "cohen_kappa_pairs": {
                    k: round(v, 4) if v is not None else None
                    for k, v in cov_kappas.items()
                },
            },
        }

        if macro_f1 is not None:
            f1_values.append(macro_f1)
        if cov_alpha is not None:
            cov_alpha_values.append(cov_alpha)
        for v in cov_kappas.values():
            if v is not None:
                cov_kappa_values.append(v)

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    results["summary"] = {
        "span_matching": {
            "mean_f1_all_labels": safe_mean(f1_values),
        },
        "coverage_agreement": {
            "mean_krippendorff_alpha":    safe_mean(cov_alpha_values),
            "mean_cohen_kappa_all_pairs": safe_mean(cov_kappa_values),
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
    print(f"  Span matching       - mean F1       : {s['span_matching']['mean_f1_all_labels']}")
    print(f"  Coverage agreement  - mean α        : {s['coverage_agreement']['mean_krippendorff_alpha']}")
    print(f"  Coverage agreement  - mean κ         : {s['coverage_agreement']['mean_cohen_kappa_all_pairs']}")
    print()

    col_w = 34
    print(f"{'Label':<{col_w}} {'F1':>8} {'cov α':>8} {'cov κ':>8}")
    print("-" * 65)
    for label, v in report["per_label"].items():
        f1    = v["span_matching"]["macro_f1"]
        cov_a = v["coverage_agreement"]["krippendorff_alpha"]
        cov_kappas = v["coverage_agreement"]["cohen_kappa_pairs"]
        mean_kappa = (
            round(sum(x for x in cov_kappas.values() if x is not None) /
                  len([x for x in cov_kappas.values() if x is not None]), 4)
            if any(x is not None for x in cov_kappas.values()) else None
        )
        f1_str    = f"{f1:>8.4f}" if f1 is not None else "     N/A"
        cov_str   = f"{cov_a:>8.4f}" if cov_a is not None else "     N/A"
        kappa_str = f"{mean_kappa:>8.4f}" if mean_kappa is not None else "     N/A"
        print(f"{label:<{col_w}}{f1_str}{cov_str}{kappa_str}")


if __name__ == "__main__":
    main()