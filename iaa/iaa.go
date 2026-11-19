package iaa

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
	ConfidenceRating int          `json:"confidence_rating"`
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
	Labelset Labelset `json:"labelset"`
	// DocumentLabelset is present on a hybrid task: labels applied to a whole
	// document alongside the span labels in Labelset.
	DocumentLabelset Labelset   `json:"document_labelset"`
	Documents        []Document `json:"documents"`
	AnnotationLevel  string     `json:"annotation_level"`
	// SpanGranularity is the same value under the name Lawnotation's download
	// export now writes. See annotationLevelOf.
	SpanGranularity string `json:"span_granularity"`
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

type CoverageAgreement struct {
	MatrixItems       int         `json:"matrix_items"`
	KrippendorffAlpha NullFloat64 `json:"krippendorff_alpha"`
	// MarkedItems counts the units at least one annotator marked. Over a
	// corpus most units are blank, and agreed-upon blankness dominates the
	// matrix, so alpha on its own can report mostly how much text nobody
	// touched. Reporting the share alongside it says how much of the score
	// that is, without inventing a second coefficient to compare against.
	MarkedItems int `json:"marked_items"`
	// AlphaPairs is the same coefficient restricted to each pair of
	// annotators. Krippendorff rather than Cohen so a pair's figure and the
	// all-annotator figure above it are the same quantity under the same
	// chance model, and so a pair where one annotator saw no documents in
	// common with the other still resolves instead of failing.
	AlphaPairs map[string]NullFloat64 `json:"krippendorff_alpha_pairs"`
	// CohenKappaPairs is the other conventional pairwise coefficient. On this
	// data it tracks AlphaPairs to within 0.003 (see coefficients_test.go);
	// it is reported because a task with exactly two annotators has no use
	// for a multi-annotator coefficient and kappa is what readers expect.
	CohenKappaPairs map[string]NullFloat64 `json:"cohen_kappa_pairs"`
	// PerDocument covers only the documents in which the label was used, so a
	// corpus figure can be read against the spread behind it.
	DocumentsWithLabel int             `json:"documents_with_label"`
	PerDocument        []DocumentAlpha `json:"per_document"`
}

type LabelResult struct {
	SpanCountsPerAnnotator map[string]int    `json:"span_counts_per_annotator"`
	SpanAgreement          SpanAgreement     `json:"span_agreement"`
	CoverageAgreement      CoverageAgreement `json:"coverage_agreement"`
}

type Meta struct {
	InputFile       string   `json:"input_file"`
	AnnotationLevel string   `json:"annotation_level"`
	Annotators      []string `json:"annotators"`
	NumDocuments    int      `json:"num_documents"`
}

// TaskData is one loaded export, ready to compute from.
type TaskData struct {
	Labels []string
	// DocumentLabels is empty unless the task is hybrid.
	DocumentLabels  []string
	Annotators      []string
	Documents       []Document
	AnnotationLevel string
}

type Report struct {
	Meta     Meta                   `json:"meta"`
	PerLabel map[string]LabelResult `json:"per_label"`
	// PerDocumentLabel is present on a hybrid task, scoring the whole-document
	// labels the way a document-level task scores its only labelset.
	PerDocumentLabel map[string]LabelResult `json:"per_document_label,omitempty"`
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
			key := fmt.Sprintf("%s_vs_%s", annotators[i], annotators[j])
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
			key := fmt.Sprintf("%s_vs_%s", annotators[i], annotators[j])
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

// tokenize splits text into the word units coverage agreement counts.
func tokenize(text string) []span {
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

func buildCoverageMatrix(documents []Document, label string, annotators []string) [][]*float64 {
	rows := make([][]*float64, len(annotators))
	for i := range rows {
		rows[i] = []*float64{}
	}
	for _, doc := range documents {
		tokens := tokenize(doc.FullText)
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
// Confidence rating agreement
// ---------------------------------------------------------------------------

// buildConfidenceMatrix returns an annotators x documents reliability matrix
// of confidence_rating values. A value of 0 (unrated) or a missing assignment
// is represented as nil (missing), matching buildCoverageMatrix's convention.
func buildConfidenceMatrix(documents []Document, annotators []string) [][]*float64 {
	rows := make([][]*float64, len(annotators))
	for i := range rows {
		rows[i] = make([]*float64, len(documents))
	}
	for docIdx, doc := range documents {
		ratings := map[string]int{}
		for _, asgn := range doc.Assignments {
			ratings[fmt.Sprintf("%v", asgn.Annotator)] = asgn.ConfidenceRating
		}
		for i, annotator := range annotators {
			if r, ok := ratings[annotator]; ok && r >= 1 && r <= 5 {
				rows[i][docIdx] = ptr(float64(r))
			}
		}
	}
	return rows
}

type ConfidenceRatingSummary struct {
	Total      int                    `json:"total"`
	Rated      int                    `json:"rated"`
	Mean       NullFloat64            `json:"mean"`
	Counts     map[int]int            `json:"counts"` // keys 1..5
	Alpha      NullFloat64            `json:"krippendorff_alpha"`
	AlphaPairs map[string]NullFloat64 `json:"krippendorff_alpha_pairs"`
	// RatedPerAnnotator says how many of each annotator's own assignments
	// carry a rating. An annotator who rated none makes every pair involving
	// them undefined, which is worth saying outright rather than leaving the
	// reader to infer it from a row of blanks.
	RatedPerAnnotator map[string]int `json:"rated_per_annotator"`
}

// confidenceRatingSummary computes total/rated counts, the star histogram,
// mean rating (over rated assignments only), and Krippendorff's alpha (both
// overall and per annotator pair) for the confidence_rating field over the
// given documents.
func confidenceRatingSummary(documents []Document, annotators []string) ConfidenceRatingSummary {
	total := 0
	counts := map[int]int{1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
	ratedPer := map[string]int{}
	for _, a := range annotators {
		ratedPer[a] = 0
	}
	var ratedVals []float64
	for _, doc := range documents {
		for _, asgn := range doc.Assignments {
			total++
			r := asgn.ConfidenceRating
			if r >= 1 && r <= 5 {
				counts[r]++
				ratedVals = append(ratedVals, float64(r))
				ratedPer[fmt.Sprintf("%v", asgn.Annotator)]++
			}
		}
	}
	matrix := buildConfidenceMatrix(documents, annotators)
	return ConfidenceRatingSummary{
		Total:             total,
		Rated:             len(ratedVals),
		Mean:              safeMean(ratedVals),
		Counts:            counts,
		Alpha:             krippendorffAlpha(matrix),
		AlphaPairs:        krippendorffAlphaAllPairs(matrix, annotators),
		RatedPerAnnotator: ratedPer,
	}
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

func spanCounts(documents []Document, label string, annotators []string) map[string]int {
	counts := map[string]int{}
	for _, a := range annotators {
		counts[a] = 0
	}
	for _, doc := range documents {
		for _, asgn := range doc.Assignments {
			key := fmt.Sprintf("%v", asgn.Annotator)
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
func loadData(path string) (TaskData, error) {
	f, err := os.Open(path)
	if err != nil {
		return TaskData{}, err
	}
	defer f.Close()
	return loadDataFromReader(f)
}

// loadDataFromReader reads a LawNotation JSON export from any reader (e.g. an
// HTTP request body), sharing the same extraction logic as loadData.
func loadDataFromReader(r io.Reader) (TaskData, error) {
	var raw InputData
	if err := json.NewDecoder(r).Decode(&raw); err != nil {
		return TaskData{}, err
	}
	names := func(ls Labelset) []string {
		out := make([]string, len(ls.Labels))
		for i, l := range ls.Labels {
			out[i] = l.Name
		}
		return out
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
	return TaskData{
		Labels:          names(raw.Labelset),
		DocumentLabels:  names(raw.DocumentLabelset),
		Annotators:      annotators,
		Documents:       raw.Documents,
		AnnotationLevel: annotationLevelOf(raw),
	}, nil
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

// computeIAAFromData computes the full IAA report from already-loaded data,
// so callers that already have the data in memory (e.g. an HTTP handler
// reading a request body) don't need to round-trip through a file path.
func computeIAAFromData(inputLabel string, data TaskData) Report {
	labels, annotators := data.Labels, data.Annotators
	documents, annotationLevel := data.Documents, data.AnnotationLevel
	isDocumentLevel := annotationLevel == "document"
	report := Report{
		Meta: Meta{
			InputFile:       inputLabel,
			AnnotationLevel: annotationLevel,
			Annotators:      annotators,
			NumDocuments:    len(documents),
		},
		PerLabel: map[string]LabelResult{},
	}
	for _, label := range labels {
		// A document-level task's own labelset is scored exactly the way a
		// hybrid task's second labelset is — one unit per document.
		if isDocumentLevel {
			report.PerLabel[label] = documentLabelResult(documents, label, annotators)
			continue
		}

		counts := spanCounts(documents, label, annotators)
		labelAgreement := spanAgreement(documents, []string{label}, annotators)

		covMatrix := buildCoverageMatrix(documents, label, annotators)
		covAlpha := krippendorffAlpha(covMatrix)
		covAlphaPairs := krippendorffAlphaAllPairs(covMatrix, annotators)
		covKappaPairs := cohenKappaAllPairs(covMatrix, annotators)
		matrixItems := 0
		if len(covMatrix) > 0 {
			matrixItems = len(covMatrix[0])
		}
		markedItems := markedColumnCount(covMatrix)
		perDocument := perDocumentAlpha(documents, label, annotators)
		report.PerLabel[label] = LabelResult{
			SpanCountsPerAnnotator: counts,
			SpanAgreement:          labelAgreement,
			CoverageAgreement: CoverageAgreement{
				MatrixItems:        matrixItems,
				KrippendorffAlpha:  covAlpha,
				MarkedItems:        markedItems,
				AlphaPairs:         covAlphaPairs,
				CohenKappaPairs:    covKappaPairs,
				DocumentsWithLabel: len(perDocument),
				PerDocument:        perDocument,
			},
		}
	}
	for _, label := range data.DocumentLabels {
		if report.PerDocumentLabel == nil {
			report.PerDocumentLabel = map[string]LabelResult{}
		}
		report.PerDocumentLabel[label] = documentLabelResult(documents, label, annotators)
	}
	return report
}
