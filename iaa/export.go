package iaa

// The downloadable report as a handful of tidy tables, one subject each.
//
// It replaces a folder per document and per label — 18,921 files for a
// 900-document task, each a CSV made of several stacked sections that no
// spreadsheet or dataframe reads as a table. Here every file is one
// rectangle with a header row, its grain is in its name, and a missing value
// is an empty cell rather than text, so numeric columns stay numeric.

import (
	"archive/zip"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"strconv"
	"time"
)

type column struct{ name, doc string }

type table struct {
	file, doc string
	columns   []column
	rows      [][]string
}

func (t *table) add(row ...string) { t.rows = append(t.rows, row) }

func num(v NullFloat64) string {
	if !v.Valid {
		return ""
	}
	return strconv.FormatFloat(math.Round(v.Value*10000)/10000, 'f', -1, 64)
}

func count(n int) string { return strconv.Itoa(n) }

type annotatorPair struct{ a, b string }

func (p annotatorPair) key() string { return p.a + "_vs_" + p.b }

func annotatorPairs(annotators []string) []annotatorPair {
	var out []annotatorPair
	for i := range annotators {
		for j := i + 1; j < len(annotators); j++ {
			out = append(out, annotatorPair{annotators[i], annotators[j]})
		}
	}
	return out
}

// labelGroup is one labelset's results. A hybrid task has two.
type labelGroup struct {
	kind    string
	labels  []string
	results map[string]LabelResult
}

func labelGroups(data TaskData, report Report) []labelGroup {
	kind := "span"
	if report.Meta.AnnotationLevel == "document" {
		kind = "document"
	}
	groups := []labelGroup{{kind, data.Labels, report.PerLabel}}
	if len(report.PerDocumentLabel) > 0 {
		groups = append(groups, labelGroup{"document", data.DocumentLabels, report.PerDocumentLabel})
	}
	return groups
}

func meanF1(agreement SpanAgreement) (strict, lenient NullFloat64) {
	var s, l []float64
	for _, p := range agreement.Pairs {
		if p.StrictF1.Valid {
			s = append(s, p.StrictF1.Value)
		}
		if p.LenientF1.Valid {
			l = append(l, p.LenientF1.Value)
		}
	}
	return safeMean(s), safeMean(l)
}

func labelsTable(data TaskData, report Report) table {
	t := table{file: "labels.csv", doc: "One row per label.", columns: []column{
		{"label", ""},
		{"kind", "span, or document for a tag applied to a whole document"},
		{"unit", "what alpha counts: word for span labels, document for document labels"},
		{"alpha", "Krippendorff's alpha over every annotator at once"},
		{"documents_with_label", "documents in which at least one annotator used the label"},
		{"marked_units", "units at least one annotator marked; the rest is agreed blank space"},
		{"total_units", "all units alpha was computed over"},
		{"f1_strict_mean", "strict span F1, averaged over annotator pairs; empty for document labels"},
		{"f1_lenient_mean", "lenient span F1, averaged over annotator pairs; empty for document labels"},
	}}
	for _, g := range labelGroups(data, report) {
		unit := "word"
		if g.kind == "document" {
			unit = "document"
		}
		for _, label := range g.labels {
			r := g.results[label]
			c := r.CoverageAgreement
			strict, lenient := meanF1(r.SpanAgreement)
			t.add(label, g.kind, unit, num(c.KrippendorffAlpha), count(c.DocumentsWithLabel),
				count(c.MarkedItems), count(c.MatrixItems), num(strict), num(lenient))
		}
	}
	return t
}

func pairsTable(data TaskData, report Report) table {
	t := table{file: "pairs.csv", doc: "One row per label and annotator pair.", columns: []column{
		{"label", ""},
		{"kind", "span or document"},
		{"annotator_a", ""},
		{"annotator_b", ""},
		{"alpha", "Krippendorff's alpha for just these two"},
		{"kappa", "Cohen's kappa for just these two"},
		{"f1_strict", "span F1 with no credit for a partial match"},
		{"f1_lenient", "span F1 with half credit for a partial match"},
		{"matched", "spans with the same boundaries and label"},
		{"boundary_smaller", "same label, b marked less text than a"},
		{"boundary_larger", "same label, b marked more text than a"},
		{"boundary_overlap", "same label, overlapping without either containing the other"},
		{"only_a", "spans a marked that b left untouched"},
		{"only_b", "spans b marked that a left untouched"},
		{"documents_compared", "documents both annotators worked on"},
	}}
	for _, g := range labelGroups(data, report) {
		for _, label := range g.labels {
			r := g.results[label]
			for _, p := range annotatorPairs(data.Annotators) {
				alpha, kappa := num(r.CoverageAgreement.AlphaPairs[p.key()]), num(r.CoverageAgreement.CohenKappaPairs[p.key()])
				span, ok := r.SpanAgreement.Pairs[p.key()]
				if !ok {
					t.add(label, g.kind, p.a, p.b, alpha, kappa, "", "", "", "", "", "", "", "", "")
					continue
				}
				pr := span.Profile
				t.add(label, g.kind, p.a, p.b, alpha, kappa, num(span.StrictF1), num(span.LenientF1),
					count(pr.TruePositives), count(pr.BoundarySmaller), count(pr.BoundaryLarger),
					count(pr.BoundaryOverlap), count(pr.OnlyFirst), count(pr.OnlySecond), count(span.Documents))
			}
		}
	}
	return t
}

func documentsTable(data TaskData, report Report) table {
	t := table{file: "documents.csv", doc: "One row per document and span label, for the documents the label was used in.", columns: []column{
		{"document", ""},
		{"label", ""},
		{"alpha", "Krippendorff's alpha within this document"},
		{"marked_words", "words at least one annotator marked with this label here"},
	}}
	for _, g := range labelGroups(data, report) {
		if g.kind != "span" {
			continue
		}
		for _, label := range g.labels {
			for _, d := range g.results[label].CoverageAgreement.PerDocument {
				t.add(d.Document, label, num(d.Alpha), count(d.MarkedItems))
			}
		}
	}
	return t
}

// documentPairsTable is the per-document breakdown the old export spread over
// a folder per document: which documents a given pair disagreed on most.
func documentPairsTable(data TaskData, report Report, onProgress func(current, total int, docName string)) table {
	t := table{file: "document_pairs.csv", doc: "One row per document, span label and annotator pair, where either of the two marked the label.", columns: []column{
		{"document", ""},
		{"label", ""},
		{"annotator_a", ""},
		{"annotator_b", ""},
		{"f1_strict", ""},
		{"f1_lenient", ""},
		{"matched", ""},
		{"boundary_smaller", ""},
		{"boundary_larger", ""},
		{"boundary_overlap", ""},
		{"only_a", ""},
		{"only_b", ""},
	}}
	if report.Meta.AnnotationLevel == "document" {
		return t
	}
	pairs := annotatorPairs(data.Annotators)
	for i, doc := range data.Documents {
		if onProgress != nil {
			onProgress(i+1, len(data.Documents), doc.Name)
		}
		single := []Document{doc}
		for _, label := range data.Labels {
			labels := map[string]bool{label: true}
			for _, p := range pairs {
				profile, compared := spanErrorProfile(single, labels, p.a, p.b)
				if compared == 0 || profile.total() == 0 {
					continue
				}
				a := profile.agreement(compared)
				t.add(doc.Name, label, p.a, p.b, num(a.StrictF1), num(a.LenientF1),
					count(profile.TruePositives), count(profile.BoundarySmaller), count(profile.BoundaryLarger),
					count(profile.BoundaryOverlap), count(profile.OnlyFirst), count(profile.OnlySecond))
			}
		}
	}
	return t
}

func annotationsTable(data TaskData) table {
	t := table{file: "annotations.csv", doc: "One row per annotation.", columns: []column{
		{"document", ""},
		{"annotator", ""},
		{"label", ""},
		{"kind", "span, or document for a tag with no position in the text"},
		{"start", "character offset into the document's text; empty for document tags"},
		{"end", ""},
		{"text", ""},
	}}
	for _, doc := range data.Documents {
		for _, asgn := range doc.Assignments {
			annotator := fmt.Sprintf("%v", asgn.Annotator)
			for _, a := range asgn.Annotations {
				if a.End <= a.Start {
					t.add(doc.Name, annotator, a.Label, "document", "", "", "")
					continue
				}
				t.add(doc.Name, annotator, a.Label, "span", count(a.Start), count(a.End), a.Text)
			}
		}
	}
	return t
}

func assignmentsTable(data TaskData) table {
	t := table{file: "assignments.csv", doc: "One row per document and annotator.", columns: []column{
		{"document", ""},
		{"annotator", ""},
		{"confidence_rating", "1 to 5 stars; empty when the annotator did not rate it"},
		{"annotations", "how many annotations this annotator made in this document"},
	}}
	for _, doc := range data.Documents {
		for _, asgn := range doc.Assignments {
			rating := ""
			if asgn.ConfidenceRating >= 1 && asgn.ConfidenceRating <= 5 {
				rating = count(asgn.ConfidenceRating)
			}
			t.add(doc.Name, fmt.Sprintf("%v", asgn.Annotator), rating, count(len(asgn.Annotations)))
		}
	}
	return t
}

func confidenceTable(data TaskData) table {
	t := table{file: "confidence.csv", doc: "One row per annotator, and a final row for everyone.", columns: []column{
		{"annotator", "or all"},
		{"assignments", ""},
		{"rated", "assignments carrying a rating"},
		{"mean_rating", "over rated assignments only"},
		{"star_1", ""}, {"star_2", ""}, {"star_3", ""}, {"star_4", ""}, {"star_5", ""},
	}}
	type tally struct {
		assignments int
		stars       [6]int
		ratings     []float64
	}
	row := func(name string, s tally) {
		t.add(name, count(s.assignments), count(len(s.ratings)), num(safeMean(s.ratings)),
			count(s.stars[1]), count(s.stars[2]), count(s.stars[3]), count(s.stars[4]), count(s.stars[5]))
	}
	per := map[string]*tally{}
	var all tally
	for _, a := range data.Annotators {
		per[a] = &tally{}
	}
	for _, doc := range data.Documents {
		for _, asgn := range doc.Assignments {
			s := per[fmt.Sprintf("%v", asgn.Annotator)]
			s.assignments++
			all.assignments++
			if r := asgn.ConfidenceRating; r >= 1 && r <= 5 {
				s.stars[r]++
				all.stars[r]++
				s.ratings = append(s.ratings, float64(r))
				all.ratings = append(all.ratings, float64(r))
			}
		}
	}
	for _, a := range data.Annotators {
		row(a, *per[a])
	}
	row("all", all)
	return t
}

func confidencePairsTable(data TaskData, confidence ConfidenceRatingSummary) table {
	t := table{file: "confidence_pairs.csv", doc: "One row per annotator pair.", columns: []column{
		{"annotator_a", ""},
		{"annotator_b", ""},
		{"alpha", "Krippendorff's alpha over the ratings: whether the two found the same documents hard; empty when either rated nothing"},
	}}
	for _, p := range annotatorPairs(data.Annotators) {
		t.add(p.a, p.b, num(confidence.AlphaPairs[p.key()]))
	}
	return t
}

// create stamps every entry with the time the report was made. zip.Writer's
// Create leaves it zero, which extracts as 31 December 1979.
func create(zw *zip.Writer, name string, made time.Time) (io.Writer, error) {
	return zw.CreateHeader(&zip.FileHeader{Name: name, Method: zip.Deflate, Modified: made})
}

func writeCSV(zw *zip.Writer, t table, made time.Time) error {
	f, err := create(zw, t.file, made)
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	header := make([]string, len(t.columns))
	for i, c := range t.columns {
		header[i] = c.name
	}
	if err := w.Write(header); err != nil {
		return err
	}
	if err := w.WriteAll(t.rows); err != nil {
		return err
	}
	return w.Error()
}

// writeReportZip writes the README and every table. onProgress, when given,
// reports the per-document pass, which is the only slow part.
func writeReportZip(zw *zip.Writer, data TaskData, report Report, confidence ConfidenceRatingSummary, onProgress func(current, total int, docName string)) error {
	tables := []table{
		labelsTable(data, report),
		pairsTable(data, report),
		documentsTable(data, report),
		documentPairsTable(data, report, onProgress),
		annotationsTable(data),
		assignmentsTable(data),
		confidenceTable(data),
		confidencePairsTable(data, confidence),
	}
	made := time.Now()
	readme, err := create(zw, "README.md", made)
	if err != nil {
		return err
	}
	if _, err := readme.Write(reportReadme(data, report, confidence, tables)); err != nil {
		return err
	}
	for _, t := range tables {
		if err := writeCSV(zw, t, made); err != nil {
			return fmt.Errorf("%s: %w", t.file, err)
		}
	}
	return nil
}
