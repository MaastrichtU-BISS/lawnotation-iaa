package iaa

import (
	"archive/zip"
	"bytes"
	"encoding/csv"
	"strings"
	"testing"
)

// Document tags arrive in document_annotations, apart from the spans, which
// is how Lawnotation sends a hybrid task. "Actor" is in both labelsets.
const hybridInput = `{
 "annotation_level": "hybrid",
 "labelset": {"labels": [{"name": "Actor"}]},
 "document_labelset": {"labels": [{"name": "Relevant"}, {"name": "Actor"}]},
 "documents": [
  {"name": "a.txt", "full_text": "The court ruled today", "assignments": [
   {"annotator": "u1", "confidence_rating": 2, "annotations": [{"start": 4, "end": 9, "label": "Actor", "text": "court"}], "document_annotations": [{"label": "Relevant"}]},
   {"annotator": "u2", "confidence_rating": 3, "annotations": [{"start": 4, "end": 9, "label": "Actor", "text": "court"}], "document_annotations": []}]},
  {"name": "b.txt", "full_text": "Another text here", "assignments": [
   {"annotator": "u1", "confidence_rating": 1, "annotations": [], "document_annotations": [{"label": "Relevant"}, {"label": "Actor"}]},
   {"annotator": "u2", "confidence_rating": 1, "annotations": [{"start": 0, "end": 7, "label": "Actor", "text": "Another"}], "document_annotations": [{"label": "Relevant"}]}]}
 ]}`

func loadHybrid(t *testing.T) TaskData {
	t.Helper()
	data, err := loadDataFromReader(strings.NewReader(hybridInput))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestHybridReportHasBothLabelsets(t *testing.T) {
	data := loadHybrid(t)
	if data.AnnotationLevel != "hybrid" || len(data.Labels) != 1 || len(data.DocumentLabels) != 2 {
		t.Fatalf("unexpected task: %+v", data)
	}
	report := computeIAAFromData("", data)
	if report.Meta.AnnotationLevel != "hybrid" {
		t.Fatalf("level = %q", report.Meta.AnnotationLevel)
	}
	span, ok := report.PerLabel["Actor"]
	if !ok || span.SpanAgreement.Pairs == nil {
		t.Fatal("span label should have span matching")
	}
	if got := span.SpanCountsPerAnnotator; got["u1"] != 1 || got["u2"] != 2 {
		t.Fatalf("a document tag counted as a span: %v", got)
	}
	doc, ok := report.PerDocumentLabel["Relevant"]
	if !ok {
		t.Fatal("missing document label result")
	}
	if doc.SpanAgreement.Pairs != nil {
		t.Fatal("document labels must not have span matching")
	}
	if got := doc.SpanCountsPerAnnotator; got["u1"] != 2 || got["u2"] != 1 {
		t.Fatalf("document tags not read from document_annotations: %v", got)
	}
	// u1 and u2 disagree on one of two documents for Relevant, so alpha < 1.
	if a := doc.CoverageAgreement.KrippendorffAlpha; !a.Valid || a.Value >= 1 {
		t.Fatalf("unexpected document alpha: %+v", a)
	}
	// A label name used in both labelsets is reported separately for each.
	actorTag, ok := report.PerDocumentLabel["Actor"]
	if !ok {
		t.Fatal("document label sharing a span label's name was lost")
	}
	if got := actorTag.SpanCountsPerAnnotator; got["u1"] != 1 || got["u2"] != 0 {
		t.Fatalf("a span counted as a document tag: %v", got)
	}
}

func TestNonHybridReportUnchanged(t *testing.T) {
	data, _ := loadDataFromReader(strings.NewReader(`{"labelset":{"labels":[{"name":"A"}]},"documents":[]}`))
	report := computeIAAFromData("", data)
	if report.PerDocumentLabel != nil {
		t.Fatal("non-hybrid report must not have per_document_label")
	}
}

func TestHybridExportListsBothKinds(t *testing.T) {
	data := loadHybrid(t)
	report := computeIAAFromData("", data)
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	if err := writeReportZip(zw, data, report, confidenceRatingSummary(data.Documents, data.Annotators), nil); err != nil {
		t.Fatal(err)
	}
	zw.Close()
	zr, _ := zip.NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	rows := func(name string) [][]string {
		for _, f := range zr.File {
			if f.Name == name {
				r, _ := f.Open()
				defer r.Close()
				all, err := csv.NewReader(r).ReadAll()
				if err != nil {
					t.Fatal(err)
				}
				return all[1:]
			}
		}
		t.Fatalf("missing %s", name)
		return nil
	}
	kinds := map[string]int{}
	for _, row := range rows("labels.csv") {
		kinds[row[0]+"/"+row[1]]++
	}
	for _, want := range []string{"Actor/span", "Actor/document", "Relevant/document"} {
		if kinds[want] != 1 {
			t.Errorf("labels.csv: want one %s row, have %v", want, kinds)
		}
	}
	tags := 0
	for _, row := range rows("annotations.csv") {
		if row[3] == "document" {
			tags++
		}
	}
	if tags != 4 {
		t.Errorf("annotations.csv lists %d document tags, want 4", tags)
	}
}
