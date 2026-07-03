// Inter-Annotator Agreement for Span Annotations
//
// Reports TWO complementary views of agreement, computed per label:
//
//  1. SPAN MATCHING  (Precision / Recall / F1)
//     "If one annotator is treated as the reference, how accurately can
//     another annotator reproduce their annotations?"
//
//  2. COVERAGE AGREEMENT  (Krippendorff's alpha, Cohen's kappa — char/word)
//     "How much text do annotators agree on?"
//
// Usage:
//
//	go run iaa.go --input data.json [--criterion exact|contained] [--granularity char|word] [--output report.json]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// Input schema
// ---------------------------------------------------------------------------

type Annotation struct {
	Start int    `json:"start"`
	End   int    `json:"end"`
	Label string `json:"label"`
}

type Assignment struct {
	Annotator   interface{}  `json:"annotator"` // int or string in the wild
	Annotations []Annotation `json:"annotations"`
}

type Document struct {
	FullText    string       `json:"full_text"`
	Assignments []Assignment `json:"assignments"`
}

type Label struct {
	Name string `json:"name"`
}

type Labelset struct {
	Labels []Label `json:"labels"`
}

type InputData struct {
	Labelset  Labelset   `json:"labelset"`
	Documents []Document `json:"documents"`
}

// ---------------------------------------------------------------------------
// Nullable float for JSON output
// ---------------------------------------------------------------------------

type NullFloat64 struct {
	Valid bool
	Value float64
}

func (n NullFloat64) MarshalJSON() ([]byte, error) {
	if !n.Valid {
		return []byte("null"), nil
	}
	return json.Marshal(math.Round(n.Value*10000) / 10000)
}

func nullFloat(v float64) NullFloat64 { return NullFloat64{Valid: true, Value: v} }
func noFloat() NullFloat64            { return NullFloat64{Valid: false} }
func roundFloat(v float64) NullFloat64 {
	return NullFloat64{Valid: true, Value: math.Round(v*10000) / 10000}
}

// ---------------------------------------------------------------------------
// Krippendorff's alpha
// ---------------------------------------------------------------------------

// krippendorffAlpha computes Krippendorff's alpha with the nominal/binary
// distance metric from a reliability matrix (rows=annotators, cols=items).
// nil entries mean the annotator was not assigned to that document.
func krippendorffAlpha(matrix [][]*float64) NullFloat64 {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return noFloat()
	}
	nAnnotators := len(matrix)
	nItems := len(matrix[0])

	doNum, doDen := 0.0, 0.0
	var allVals []float64

	for col := 0; col < nItems; col++ {
		var colVals []float64
		for r := 0; r < nAnnotators; r++ {
			if matrix[r][col] != nil {
				colVals = append(colVals, *matrix[r][col])
			}
		}
		allVals = append(allVals, colVals...)
		mU := len(colVals)
		if mU < 2 {
			continue
		}
		for i := 0; i < mU; i++ {
			for j := i + 1; j < mU; j++ {
				if colVals[i] != colVals[j] {
					doNum += 1.0
				}
				doDen += 1.0
			}
		}
	}

	if doDen == 0 {
		return noFloat()
	}
	Do := doNum / doDen
	n := float64(len(allVals))
	if n < 2 {
		return noFloat()
	}

	c1 := 0.0
	for _, v := range allVals {
		c1 += v
	}
	c0 := n - c1
	De := 2.0 * c1 * c0 / (n * (n - 1))

	if De == 0.0 {
		if Do == 0.0 {
			return nullFloat(1.0)
		}
		return noFloat()
	}
	return nullFloat(1.0 - Do/De)
}

// ---------------------------------------------------------------------------
// Cohen's kappa
// ---------------------------------------------------------------------------

func cohenKappaPair(valsA, valsB []*float64) NullFloat64 {
	type pair struct{ a, b float64 }
	var paired []pair
	for i := range valsA {
		if valsA[i] != nil && valsB[i] != nil {
			paired = append(paired, pair{*valsA[i], *valsB[i]})
		}
	}
	if len(paired) < 2 {
		return noFloat()
	}
	n := float64(len(paired))
	pO, pA, pB := 0.0, 0.0, 0.0
	for _, p := range paired {
		if p.a == p.b {
			pO += 1.0
		}
		pA += p.a
		pB += p.b
	}
	pO /= n
	pA /= n
	pB /= n
	pE := pA*pB + (1-pA)*(1-pB)

	if math.Abs(pE-1.0) < 1e-9 {
		if math.Abs(pO-1.0) < 1e-9 {
			return nullFloat(1.0)
		}
		return noFloat()
	}
	return nullFloat((pO - pE) / (1.0 - pE))
}

func cohenKappaAllPairs(matrix [][]*float64, annotators []string) map[string]NullFloat64 {
	result := map[string]NullFloat64{}
	for i := 0; i < len(annotators); i++ {
		for j := i + 1; j < len(annotators); j++ {
			key := fmt.Sprintf("annotator_%s_vs_annotator_%s", annotators[i], annotators[j])
			result[key] = cohenKappaPair(matrix[i], matrix[j])
		}
	}
	return result
}

// ---------------------------------------------------------------------------
// View 2 — Coverage agreement
// ---------------------------------------------------------------------------

type span struct{ start, end int }

var wordRe = regexp.MustCompile(`\S+`)

func tokenize(text, granularity string) []span {
	if granularity == "char" {
		// iterate over Unicode code points (runes) for correct char offsets
		tokens := make([]span, 0, len(text))
		for i, r := range text {
			tokens = append(tokens, span{i, i + len(string(r))})
		}
		return tokens
	}
	locs := wordRe.FindAllStringIndex(text, -1)
	tokens := make([]span, len(locs))
	for i, loc := range locs {
		tokens[i] = span{loc[0], loc[1]}
	}
	return tokens
}

func tokenIsCovered(tokStart, tokEnd int, annotations []Annotation, label string) bool {
	for _, ann := range annotations {
		if ann.Label == label && ann.Start <= tokStart && ann.End >= tokEnd {
			return true
		}
	}
	return false
}

func ptr(v float64) *float64 { p := v; return &p }

func buildCoverageMatrix(documents []Document, label string, annotators []string, granularity string) [][]*float64 {
	rows := make([][]*float64, len(annotators))
	for i := range rows {
		rows[i] = []*float64{}
	}

	for _, doc := range documents {
		tokens := tokenize(doc.FullText, granularity)
		annMap := buildAnnMap(doc)

		for _, tok := range tokens {
			for i, annotator := range annotators {
				anns, assigned := annMap[annotator]
				if !assigned {
					rows[i] = append(rows[i], nil)
				} else {
					if tokenIsCovered(tok.start, tok.end, anns, label) {
						rows[i] = append(rows[i], ptr(1.0))
					} else {
						rows[i] = append(rows[i], ptr(0.0))
					}
				}
			}
		}
	}
	return rows
}

// buildAnnMap converts doc assignments into annotator-id -> annotations map.
func buildAnnMap(doc Document) map[string][]Annotation {
	m := map[string][]Annotation{}
	for _, asgn := range doc.Assignments {
		key := fmt.Sprintf("%v", asgn.Annotator)
		m[key] = asgn.Annotations
	}
	return m
}

// ---------------------------------------------------------------------------
// View 1 — Span matching
// ---------------------------------------------------------------------------

func spansMatch(a, b span, criterion string) bool {
	if criterion == "exact" {
		return a.start == b.start && a.end == b.end
	}
	// contained: one must be a subset of the other
	return (a.start >= b.start && a.end <= b.end) || (b.start >= a.start && b.end <= a.end)
}

// matchSpanSets finds the maximum bipartite matching using augmenting-path DFS.
// Returns (truePositives, refUnmatched, sysUnmatched).
func matchSpanSets(spansRef, spansSys []span, criterion string) (int, int, int) {
	// build adjacency list: adj[i] = list of sys indices that match ref[i]
	adj := make([][]int, len(spansRef))
	for i, rs := range spansRef {
		for j, ss := range spansSys {
			if spansMatch(rs, ss, criterion) {
				adj[i] = append(adj[i], j)
			}
		}
	}

	matchSys := make([]int, len(spansSys))
	for i := range matchSys {
		matchSys[i] = -1
	}

	var augment func(i int, visited map[int]bool) bool
	augment = func(i int, visited map[int]bool) bool {
		for _, j := range adj[i] {
			if visited[j] {
				continue
			}
			visited[j] = true
			if matchSys[j] == -1 || augment(matchSys[j], visited) {
				matchSys[j] = i
				return true
			}
		}
		return false
	}

	tp := 0
	for i := range spansRef {
		if augment(i, map[int]bool{}) {
			tp++
		}
	}
	return tp, len(spansRef) - tp, len(spansSys) - tp
}

// PairResult mirrors the Python precision_recall_f1 return value.
type PairResult struct {
	TruePositives     int         `json:"true_positives"`
	RefSpanCount      int         `json:"ref_span_count"`
	SysSpanCount      int         `json:"sys_span_count"`
	Precision         NullFloat64 `json:"precision"`
	Recall            NullFloat64 `json:"recall"`
	F1                NullFloat64 `json:"f1"`
	DocumentsCompared int         `json:"documents_compared"`
}

func precisionRecallF1(documents []Document, label, annotatorRef, annotatorSys, criterion string) PairResult {
	totalTP, totalRefUn, totalSysUn, nDocs := 0, 0, 0, 0

	for _, doc := range documents {
		annMap := buildAnnMap(doc)
		refAnns, hasRef := annMap[annotatorRef]
		sysAnns, hasSys := annMap[annotatorSys]
		if !hasRef || !hasSys {
			continue
		}
		nDocs++

		var spansRef, spansSys []span
		for _, a := range refAnns {
			if a.Label == label {
				spansRef = append(spansRef, span{a.Start, a.End})
			}
		}
		for _, a := range sysAnns {
			if a.Label == label {
				spansSys = append(spansSys, span{a.Start, a.End})
			}
		}

		tp, refUn, sysUn := matchSpanSets(spansRef, spansSys, criterion)
		totalTP += tp
		totalRefUn += refUn
		totalSysUn += sysUn
	}

	nRef := totalTP + totalRefUn
	nSys := totalTP + totalSysUn

	var precision, recall, f1 NullFloat64
	if nSys > 0 {
		precision = roundFloat(float64(totalTP) / float64(nSys))
	}
	if nRef > 0 {
		recall = roundFloat(float64(totalTP) / float64(nRef))
	}

	if precision.Valid && recall.Valid && (precision.Value+recall.Value) > 0 {
		f1 = roundFloat(2 * precision.Value * recall.Value / (precision.Value + recall.Value))
	} else if (precision.Valid && precision.Value == 0) || (recall.Valid && recall.Value == 0) {
		f1 = nullFloat(0.0)
	}

	return PairResult{
		TruePositives:     totalTP,
		RefSpanCount:      nRef,
		SysSpanCount:      nSys,
		Precision:         precision,
		Recall:            recall,
		F1:                f1,
		DocumentsCompared: nDocs,
	}
}

type DirectionPair struct {
	AAsRef PairResult `json:"-"` // keys set dynamically
	BAsRef PairResult `json:"-"`
}

func spanMatchingAllPairs(documents []Document, label string, annotators []string, criterion string) map[string]map[string]PairResult {
	results := map[string]map[string]PairResult{}
	for i := 0; i < len(annotators); i++ {
		for j := i + 1; j < len(annotators); j++ {
			a, b := annotators[i], annotators[j]
			key := fmt.Sprintf("annotator_%s_vs_annotator_%s", a, b)
			results[key] = map[string]PairResult{
				fmt.Sprintf("annotator_%s_as_reference", a): precisionRecallF1(documents, label, a, b, criterion),
				fmt.Sprintf("annotator_%s_as_reference", b): precisionRecallF1(documents, label, b, a, criterion),
			}
		}
	}
	return results
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

func spanCounts(documents []Document, label string, annotators []string) map[string]int {
	counts := map[string]int{}
	for _, a := range annotators {
		counts[fmt.Sprintf("annotator_%s", a)] = 0
	}
	for _, doc := range documents {
		for _, asgn := range doc.Assignments {
			key := fmt.Sprintf("annotator_%v", asgn.Annotator)
			if _, ok := counts[key]; ok {
				for _, ann := range asgn.Annotations {
					if ann.Label == label {
						counts[key]++
					}
				}
			}
		}
	}
	return counts
}

func loadData(path string) ([]string, []string, []Document, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()

	var raw InputData
	if err := json.NewDecoder(f).Decode(&raw); err != nil {
		return nil, nil, nil, err
	}

	labels := make([]string, len(raw.Labelset.Labels))
	for i, l := range raw.Labelset.Labels {
		labels[i] = l.Name
	}

	annotatorSet := map[string]bool{}
	for _, doc := range raw.Documents {
		for _, asgn := range doc.Assignments {
			annotatorSet[fmt.Sprintf("%v", asgn.Annotator)] = true
		}
	}
	annotators := make([]string, 0, len(annotatorSet))
	for a := range annotatorSet {
		annotators = append(annotators, a)
	}
	sort.Strings(annotators)

	return labels, annotators, raw.Documents, nil
}

// ---------------------------------------------------------------------------
// Output schema
// ---------------------------------------------------------------------------

type CoverageAgreement struct {
	MatrixItems       int                    `json:"matrix_items"`
	KrippendorffAlpha NullFloat64            `json:"krippendorff_alpha"`
	CohenKappaPairs   map[string]NullFloat64 `json:"cohen_kappa_pairs"`
}

type SpanMatchingSummary struct {
	MacroF1 NullFloat64                      `json:"macro_f1"`
	Pairs   map[string]map[string]PairResult `json:"pairs"`
}

type LabelResult struct {
	SpanCountsPerAnnotator map[string]int      `json:"span_counts_per_annotator"`
	SpanMatching           SpanMatchingSummary `json:"span_matching"`
	CoverageAgreement      CoverageAgreement   `json:"coverage_agreement"`
}

type SummarySpanMatching struct {
	MeanF1AllLabels NullFloat64 `json:"mean_f1_all_labels"`
}

type SummaryCoverage struct {
	MeanKrippendorffAlpha  NullFloat64 `json:"mean_krippendorff_alpha"`
	MeanCohenKappaAllPairs NullFloat64 `json:"mean_cohen_kappa_all_pairs"`
}

type Summary struct {
	SpanMatching        SummarySpanMatching `json:"span_matching"`
	CoverageAgreement   SummaryCoverage     `json:"coverage_agreement"`
	InterpretationGuide map[string]string   `json:"interpretation_guide_alpha_kappa"`
}

type Meta struct {
	InputFile    string            `json:"input_file"`
	Criterion    string            `json:"criterion"`
	Granularity  string            `json:"granularity"`
	Annotators   []string          `json:"annotators"`
	NumDocuments int               `json:"num_documents"`
	Notes        map[string]string `json:"notes"`
}

type Report struct {
	Meta     Meta                   `json:"meta"`
	PerLabel map[string]LabelResult `json:"per_label"`
	Summary  Summary                `json:"summary"`
}

// ---------------------------------------------------------------------------
// Main computation
// ---------------------------------------------------------------------------

func safeMean(vals []float64) NullFloat64 {
	if len(vals) == 0 {
		return noFloat()
	}
	s := 0.0
	for _, v := range vals {
		s += v
	}
	return roundFloat(s / float64(len(vals)))
}

func computeIAA(inputPath, criterion, granularity string) (Report, error) {
	labels, annotators, documents, err := loadData(inputPath)
	if err != nil {
		return Report{}, err
	}

	prefixedAnnotators := make([]string, len(annotators))
	for i, a := range annotators {
		prefixedAnnotators[i] = "annotator_" + a
	}

	report := Report{
		Meta: Meta{
			InputFile:    inputPath,
			Criterion:    criterion,
			Granularity:  granularity,
			Annotators:   prefixedAnnotators,
			NumDocuments: len(documents),
			Notes: map[string]string{
				"span_matching": fmt.Sprintf(
					"Precision/Recall/F1 from matching whole spans between "+
						"annotator pairs, using criterion='%s'. Not "+
						"chance-corrected; reported per-pair in both directions "+
						"since precision/recall are asymmetric. Answers: if one "+
						"annotator is the reference, how accurately can another "+
						"reproduce their annotations?", criterion),
				"coverage_agreement": fmt.Sprintf(
					"Krippendorff's alpha / Cohen's kappa on a "+
						"%s-level reliability matrix. 1=token covered "+
						"by label, 0=not covered, None=doc not assigned. "+
						"LENGTH-WEIGHTED: agreement on long spans contributes more "+
						"items than agreement on short spans. Answers: how much "+
						"text do annotators agree on?", granularity),
			},
		},
		PerLabel: map[string]LabelResult{},
	}

	var covAlphaVals, covKappaVals, f1Vals []float64

	for _, label := range labels {
		counts := spanCounts(documents, label, annotators)

		// View 1: span matching
		matching := spanMatchingAllPairs(documents, label, annotators, criterion)
		var labelF1s []float64
		for _, pairResult := range matching {
			for _, dir := range pairResult {
				if dir.F1.Valid {
					labelF1s = append(labelF1s, dir.F1.Value)
				}
			}
		}
		macroF1 := safeMean(labelF1s)

		// View 2: coverage agreement
		covMatrix := buildCoverageMatrix(documents, label, annotators, granularity)
		covAlpha := krippendorffAlpha(covMatrix)
		covKappas := cohenKappaAllPairs(covMatrix, annotators)

		matrixItems := 0
		if len(covMatrix) > 0 {
			matrixItems = len(covMatrix[0])
		}

		report.PerLabel[label] = LabelResult{
			SpanCountsPerAnnotator: counts,
			SpanMatching: SpanMatchingSummary{
				MacroF1: macroF1,
				Pairs:   matching,
			},
			CoverageAgreement: CoverageAgreement{
				MatrixItems:       matrixItems,
				KrippendorffAlpha: covAlpha,
				CohenKappaPairs:   covKappas,
			},
		}

		if macroF1.Valid {
			f1Vals = append(f1Vals, macroF1.Value)
		}
		if covAlpha.Valid {
			covAlphaVals = append(covAlphaVals, covAlpha.Value)
		}
		for _, v := range covKappas {
			if v.Valid {
				covKappaVals = append(covKappaVals, v.Value)
			}
		}
	}

	report.Summary = Summary{
		SpanMatching: SummarySpanMatching{
			MeanF1AllLabels: safeMean(f1Vals),
		},
		CoverageAgreement: SummaryCoverage{
			MeanKrippendorffAlpha:  safeMean(covAlphaVals),
			MeanCohenKappaAllPairs: safeMean(covKappaVals),
		},
		InterpretationGuide: map[string]string{
			"< 0.20":    "Slight agreement",
			"0.21-0.40": "Fair agreement",
			"0.41-0.60": "Moderate agreement",
			"0.61-0.80": "Substantial agreement",
			"> 0.80":    "Almost perfect agreement",
		},
	}

	return report, nil
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

func main() {
	input := flag.String("input", "", "Path to LawNotation JSON export (required)")
	criterion := flag.String("criterion", "exact", "Match criterion: exact|contained")
	granularity := flag.String("granularity", "word", "Coverage granularity: char|word")
	output := flag.String("output", "iaa_report.json", "Output JSON report path")
	flag.Parse()

	if *input == "" {
		fmt.Fprintln(os.Stderr, "error: --input is required")
		flag.Usage()
		os.Exit(1)
	}
	if *criterion != "exact" && *criterion != "contained" {
		fmt.Fprintln(os.Stderr, "error: --criterion must be 'exact' or 'contained'")
		os.Exit(1)
	}
	if *granularity != "char" && *granularity != "word" {
		fmt.Fprintln(os.Stderr, "error: --granularity must be 'char' or 'word'")
		os.Exit(1)
	}

	fmt.Printf("Input        : %s\n", *input)
	fmt.Printf("Criterion    : %s\n", *criterion)
	fmt.Printf("Granularity  : %s\n", *granularity)
	fmt.Printf("Output       : %s\n\n", *output)

	report, err := computeIAA(*input, *criterion, *granularity)
	if err != nil {
		log.Fatalf("error: %v", err)
	}

	f, err := os.Create(*output)
	if err != nil {
		log.Fatalf("error creating output file: %v", err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		log.Fatalf("error writing JSON: %v", err)
	}

	fmt.Printf("Report written to: %s\n\n", *output)

	s := report.Summary
	fmt.Println("=== SUMMARY ===")
	fmt.Printf("  Documents                          : %d\n", report.Meta.NumDocuments)
	fmt.Printf("  Annotators                         : %s\n", strings.Join(report.Meta.Annotators, ", "))
	fmt.Printf("  Granularity (coverage view)        : %s\n", report.Meta.Granularity)
	fmt.Printf("  Span matching       - mean F1       : %v\n", s.SpanMatching.MeanF1AllLabels)
	fmt.Printf("  Coverage agreement  - mean α        : %v\n", s.CoverageAgreement.MeanKrippendorffAlpha)
	fmt.Printf("  Coverage agreement  - mean κ        : %v\n\n", s.CoverageAgreement.MeanCohenKappaAllPairs)

	colW := 34
	fmt.Printf("%-*s %8s %8s %8s\n", colW, "Label", "F1", "cov α", "cov κ")
	fmt.Println(strings.Repeat("-", 65))

	// print in label order
	for _, label := range report.Meta.Annotators { // reuse label order from labels slice
		_ = label
	}
	// iterate over per_label in insertion order via the original labels slice
	// (re-derive from meta notes order isn't available; iterate map sorted)
	labelNames := make([]string, 0, len(report.PerLabel))
	for k := range report.PerLabel {
		labelNames = append(labelNames, k)
	}
	sort.Strings(labelNames)

	for _, label := range labelNames {
		v := report.PerLabel[label]
		f1Val := v.SpanMatching.MacroF1
		covA := v.CoverageAgreement.KrippendorffAlpha
		kappas := v.CoverageAgreement.CohenKappaPairs
		var kVals []float64
		for _, kv := range kappas {
			if kv.Valid {
				kVals = append(kVals, kv.Value)
			}
		}
		meanKappa := safeMean(kVals)

		f1Str, covStr, kappaStr := "     N/A", "     N/A", "     N/A"
		if f1Val.Valid {
			f1Str = fmt.Sprintf("%8.4f", f1Val.Value)
		}
		if covA.Valid {
			covStr = fmt.Sprintf("%8.4f", covA.Value)
		}
		if meanKappa.Valid {
			kappaStr = fmt.Sprintf("%8.4f", meanKappa.Value)
		}
		fmt.Printf("%-*s%s%s%s\n", colW, label, f1Str, covStr, kappaStr)
	}
}
