package main

// iaa.go — IAA computation logic (types, metrics, data loading).
// The CLI entry point and output formatting live in iaa.go.

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"sort"
)

// ---------------------------------------------------------------------------
// Input schema
// ---------------------------------------------------------------------------

type Annotation struct {
	Start int    `json:"start"`
	End   int    `json:"end"`
	Label string `json:"label"`
	Text  string `json:"text"`
}

type Assignment struct {
	Annotator        interface{}  `json:"annotator"` // int or string in the wild
	DifficultyRating int          `json:"difficulty_rating"`
	Annotations      []Annotation `json:"annotations"`
}

type Document struct {
	Name        string       `json:"name"`
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
	Labelset        Labelset   `json:"labelset"`
	Documents       []Document `json:"documents"`
	AnnotationLevel string     `json:"annotation_level"`
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
// Output schema
// ---------------------------------------------------------------------------

type PairResult struct {
	TruePositives     int         `json:"true_positives"`
	RefSpanCount      int         `json:"ref_span_count"`
	SysSpanCount      int         `json:"sys_span_count"`
	Precision         NullFloat64 `json:"precision"`
	Recall            NullFloat64 `json:"recall"`
	F1                NullFloat64 `json:"f1"`
	DocumentsCompared int         `json:"documents_compared"`
}

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

type Meta struct {
	InputFile       string            `json:"input_file"`
	AnnotationLevel string            `json:"annotation_level"`
	Criterion       string            `json:"criterion"`
	Granularity     string            `json:"granularity"`
	Annotators      []string          `json:"annotators"`
	NumDocuments    int               `json:"num_documents"`
	Notes           map[string]string `json:"notes"`
}

type Report struct {
	Meta     Meta                   `json:"meta"`
	PerLabel map[string]LabelResult `json:"per_label"`
}

// ---------------------------------------------------------------------------
// Krippendorff's alpha
// ---------------------------------------------------------------------------

// krippendorffAlpha computes Krippendorff's alpha using the interval/quadratic
// distance function delta(a,b) = (a-b)^2. Alpha is invariant to any constant
// scaling of delta, so this is equivalent to a normalized quadratic weight
// (e.g. 1 - d^2/(q-1)^2) without needing to hardcode the category count q.
// For binary (0/1) data — the coverage-agreement view — this reduces exactly
// to the nominal metric, since (0-1)^2 = 1 and (0-0)^2 = (1-1)^2 = 0.
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
		mU := len(colVals)
		if mU < 2 {
			// Unpairable unit (fewer than 2 raters): excluded entirely,
			// including from the expected-agreement baseline below.
			continue
		}
		allVals = append(allVals, colVals...)
		for i := 0; i < mU; i++ {
			for j := i + 1; j < mU; j++ {
				d := colVals[i] - colVals[j]
				doNum += d * d
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

	catCounts := map[float64]float64{}
	for _, v := range allVals {
		catCounts[v]++
	}
	cats := make([]float64, 0, len(catCounts))
	for c := range catCounts {
		cats = append(cats, c)
	}
	deSum := 0.0
	for _, c := range cats {
		for _, k := range cats {
			if c == k {
				continue
			}
			d := c - k
			deSum += catCounts[c] * catCounts[k] * d * d
		}
	}
	De := deSum / (n * (n - 1))

	if De == 0.0 {
		if Do == 0.0 {
			return nullFloat(1.0)
		}
		return noFloat()
	}
	return nullFloat(1.0 - Do/De)
}

// krippendorffAlphaAllPairs computes Krippendorff's alpha restricted to each
// pair of annotators in turn (mirroring cohenKappaAllPairs).
func krippendorffAlphaAllPairs(matrix [][]*float64, annotators []string) map[string]NullFloat64 {
	result := map[string]NullFloat64{}
	for i := 0; i < len(annotators); i++ {
		for j := i + 1; j < len(annotators); j++ {
			key := fmt.Sprintf("annotator_%s_vs_annotator_%s", annotators[i], annotators[j])
			result[key] = krippendorffAlpha([][]*float64{matrix[i], matrix[j]})
		}
	}
	return result
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

func buildAnnMap(doc Document) map[string][]Annotation {
	m := map[string][]Annotation{}
	for _, asgn := range doc.Assignments {
		key := fmt.Sprintf("%v", asgn.Annotator)
		m[key] = asgn.Annotations
	}
	return m
}

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

// buildPresenceMatrix returns an annotators x documents reliability matrix
// for document-level annotation data: 1.0 if that annotator applied the given
// label anywhere in the document, 0.0 if assigned to the document but didn't,
// nil if the annotator wasn't assigned to that document at all. Unlike
// buildCoverageMatrix, this doesn't tokenize text — annotations at this level
// have no meaningful span extent (start/end are always 0).
func buildPresenceMatrix(documents []Document, label string, annotators []string) [][]*float64 {
	rows := make([][]*float64, len(annotators))
	for i := range rows {
		rows[i] = make([]*float64, len(documents))
	}
	for docIdx, doc := range documents {
		annMap := buildAnnMap(doc)
		for i, annotator := range annotators {
			anns, assigned := annMap[annotator]
			if !assigned {
				continue
			}
			applied := false
			for _, ann := range anns {
				if ann.Label == label {
					applied = true
					break
				}
			}
			if applied {
				rows[i][docIdx] = ptr(1.0)
			} else {
				rows[i][docIdx] = ptr(0.0)
			}
		}
	}
	return rows
}

// ---------------------------------------------------------------------------
// Difficulty rating agreement
// ---------------------------------------------------------------------------

// buildDifficultyMatrix returns an annotators x documents reliability matrix
// of difficulty_rating values. A value of 0 (unrated) or a missing assignment
// is represented as nil (missing), matching buildCoverageMatrix's convention.
func buildDifficultyMatrix(documents []Document, annotators []string) [][]*float64 {
	rows := make([][]*float64, len(annotators))
	for i := range rows {
		rows[i] = make([]*float64, len(documents))
	}
	for docIdx, doc := range documents {
		ratings := map[string]int{}
		for _, asgn := range doc.Assignments {
			ratings[fmt.Sprintf("%v", asgn.Annotator)] = asgn.DifficultyRating
		}
		for i, annotator := range annotators {
			if r, ok := ratings[annotator]; ok && r >= 1 && r <= 5 {
				rows[i][docIdx] = ptr(float64(r))
			}
		}
	}
	return rows
}

type DifficultyRatingSummary struct {
	Total      int                    `json:"total"`
	Rated      int                    `json:"rated"`
	Mean       NullFloat64            `json:"mean"`
	Counts     map[int]int            `json:"counts"` // keys 1..5
	Alpha      NullFloat64            `json:"krippendorff_alpha"`
	AlphaPairs map[string]NullFloat64 `json:"krippendorff_alpha_pairs"`
}

// difficultyRatingSummary computes total/rated counts, the star histogram,
// mean rating (over rated assignments only), and Krippendorff's alpha (both
// overall and per annotator pair) for the difficulty_rating field over the
// given documents.
func difficultyRatingSummary(documents []Document, annotators []string) DifficultyRatingSummary {
	total := 0
	counts := map[int]int{1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
	var ratedVals []float64
	for _, doc := range documents {
		for _, asgn := range doc.Assignments {
			total++
			r := asgn.DifficultyRating
			if r >= 1 && r <= 5 {
				counts[r]++
				ratedVals = append(ratedVals, float64(r))
			}
		}
	}
	matrix := buildDifficultyMatrix(documents, annotators)
	return DifficultyRatingSummary{
		Total:      total,
		Rated:      len(ratedVals),
		Mean:       safeMean(ratedVals),
		Counts:     counts,
		Alpha:      krippendorffAlpha(matrix),
		AlphaPairs: krippendorffAlphaAllPairs(matrix, annotators),
	}
}

// ---------------------------------------------------------------------------
// View 1 — Span matching
// ---------------------------------------------------------------------------

func spansMatch(a, b span, criterion string) bool {
	if criterion == "exact" {
		return a.start == b.start && a.end == b.end
	}
	return (a.start >= b.start && a.end <= b.end) || (b.start >= a.start && b.end <= a.end)
}

// matchSpanSets finds the maximum bipartite matching using augmenting-path DFS.
func matchSpanSets(spansRef, spansSys []span, criterion string) (int, int, int) {
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

// loadData reads a LawNotation JSON export from disk (CLI path).
func loadData(path string) ([]string, []string, []Document, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, "", err
	}
	defer f.Close()
	return loadDataFromReader(f)
}

// loadDataFromReader reads a LawNotation JSON export from any reader (e.g. an
// HTTP request body), sharing the same extraction logic as loadData.
func loadDataFromReader(r io.Reader) ([]string, []string, []Document, string, error) {
	var raw InputData
	if err := json.NewDecoder(r).Decode(&raw); err != nil {
		return nil, nil, nil, "", err
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
	annotationLevel := raw.AnnotationLevel
	if annotationLevel == "" {
		annotationLevel = "span"
	}
	return labels, annotators, raw.Documents, annotationLevel, nil
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

// computeIAA loads a LawNotation JSON export from disk and computes the
// full IAA report (CLI path).
func computeIAA(inputPath, criterion, granularity string) (Report, error) {
	labels, annotators, documents, annotationLevel, err := loadData(inputPath)
	if err != nil {
		return Report{}, err
	}
	return computeIAAFromData(inputPath, labels, annotators, documents, annotationLevel, criterion, granularity), nil
}

// computeIAAFromData computes the full IAA report from already-loaded data,
// so callers that already have the data in memory (e.g. an HTTP handler
// reading a request body) don't need to round-trip through a file path.
func computeIAAFromData(inputLabel string, labels, annotators []string, documents []Document, annotationLevel, criterion, granularity string) Report {
	isDocumentLevel := annotationLevel == "document"
	prefixedAnnotators := make([]string, len(annotators))
	for i, a := range annotators {
		prefixedAnnotators[i] = "annotator_" + a
	}
	notes := map[string]string{
		"coverage_agreement": fmt.Sprintf(
			"Krippendorff's alpha / Cohen's kappa on a "+
				"%s-level reliability matrix. LENGTH-WEIGHTED.", granularity),
	}
	if isDocumentLevel {
		notes["coverage_agreement"] = "Krippendorff's alpha / Cohen's kappa on whether each " +
			"annotator applied the label to the document at all (annotation_level=document, " +
			"so spans have no extent). Span matching is not applicable and omitted."
	} else {
		notes["span_matching"] = fmt.Sprintf(
			"Precision/Recall/F1 from matching whole spans between "+
				"annotator pairs, using criterion='%s'. Not "+
				"chance-corrected; reported per-pair in both directions "+
				"since precision/recall are asymmetric.", criterion)
	}
	report := Report{
		Meta: Meta{
			InputFile:       inputLabel,
			AnnotationLevel: annotationLevel,
			Criterion:       criterion,
			Granularity:     granularity,
			Annotators:      prefixedAnnotators,
			NumDocuments:    len(documents),
			Notes:           notes,
		},
		PerLabel: map[string]LabelResult{},
	}
	for _, label := range labels {
		counts := spanCounts(documents, label, annotators)

		var macroF1 NullFloat64
		var matching map[string]map[string]PairResult
		if isDocumentLevel {
			macroF1 = noFloat()
		} else {
			matching = spanMatchingAllPairs(documents, label, annotators, criterion)
			var labelF1s []float64
			for _, pairResult := range matching {
				for _, dir := range pairResult {
					if dir.F1.Valid {
						labelF1s = append(labelF1s, dir.F1.Value)
					}
				}
			}
			macroF1 = safeMean(labelF1s)
		}

		var covMatrix [][]*float64
		if isDocumentLevel {
			covMatrix = buildPresenceMatrix(documents, label, annotators)
		} else {
			covMatrix = buildCoverageMatrix(documents, label, annotators, granularity)
		}
		covAlpha := krippendorffAlpha(covMatrix)
		covKappas := cohenKappaAllPairs(covMatrix, annotators)
		matrixItems := 0
		if len(covMatrix) > 0 {
			matrixItems = len(covMatrix[0])
		}
		report.PerLabel[label] = LabelResult{
			SpanCountsPerAnnotator: counts,
			SpanMatching:           SpanMatchingSummary{MacroF1: macroF1, Pairs: matching},
			CoverageAgreement: CoverageAgreement{
				MatrixItems:       matrixItems,
				KrippendorffAlpha: covAlpha,
				CohenKappaPairs:   covKappas,
			},
		}
	}
	return report
}
