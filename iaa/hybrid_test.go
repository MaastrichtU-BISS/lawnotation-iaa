package iaa

import (
	"archive/zip"
	"bytes"
	"strings"
	"testing"
)

const hybridInput = `{
 "annotation_level": "hybrid",
 "labelset": {"labels": [{"name": "Actor"}]},
 "document_labelset": {"labels": [{"name": "Relevant"}, {"name": "Actor"}]},
 "documents": [
  {"name": "a.txt", "full_text": "The court ruled today", "assignments": [
   {"annotator": "u1", "difficulty_rating": 2, "annotations": [{"start": 4, "end": 9, "label": "Actor", "text": "court"}], "document_annotations": [{"label": "Relevant"}]},
   {"annotator": "u2", "difficulty_rating": 3, "annotations": [{"start": 4, "end": 9, "label": "Actor", "text": "court"}], "document_annotations": []}]},
  {"name": "b.txt", "full_text": "Another text here", "assignments": [
   {"annotator": "u1", "difficulty_rating": 1, "annotations": [], "document_annotations": [{"label": "Relevant"}]},
   {"annotator": "u2", "difficulty_rating": 1, "annotations": [{"start": 0, "end": 7, "label": "Actor", "text": "Another"}], "document_annotations": [{"label": "Relevant"}]}]}
 ]}`

func TestHybridReportHasBothLabelsets(t *testing.T) {
	ds, err := loadDataFromReader(strings.NewReader(hybridInput))
	if err != nil {
		t.Fatal(err)
	}
	if !ds.IsHybrid() || len(ds.Labels) != 1 || len(ds.DocumentLabels) != 2 {
		t.Fatalf("unexpected dataset: %+v", ds)
	}
	report := computeIAAFromData("", ds, "exact", "word")
	if report.Meta.AnnotationLevel != "hybrid" {
		t.Fatalf("level = %q", report.Meta.AnnotationLevel)
	}
	span, ok := report.PerLabel["Actor"]
	if !ok || span.SpanMatching.Pairs == nil {
		t.Fatal("span label should have span matching")
	}
	doc, ok := report.PerDocumentLabel["Relevant"]
	if !ok {
		t.Fatal("missing document label result")
	}
	if doc.SpanMatching.Pairs != nil {
		t.Fatal("document labels must not have span matching")
	}
	// A label name used in both labelsets is reported separately for each.
	if _, ok := report.PerDocumentLabel["Actor"]; !ok {
		t.Fatal("document label sharing a span label's name was lost")
	}
	// u1 and u2 disagree on one of two documents for Relevant, so alpha < 1.
	if a := doc.CoverageAgreement.KrippendorffAlpha; !a.Valid || a.Value >= 1 {
		t.Fatalf("unexpected document alpha: %+v", a)
	}
}

func TestNonHybridReportUnchanged(t *testing.T) {
	ds, _ := loadDataFromReader(strings.NewReader(`{"labelset":{"labels":[{"name":"A"}]},"documents":[]}`))
	report := computeIAAFromData("", ds, "exact", "word")
	if report.PerDocumentLabel != nil {
		t.Fatal("non-hybrid report must not have per_document_label")
	}
}

func TestHybridZipLayout(t *testing.T) {
	ds, _ := loadDataFromReader(strings.NewReader(hybridInput))
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	if err := writeReportZip(zw, ds, "exact", "word", nil); err != nil {
		t.Fatal(err)
	}
	zw.Close()
	zr, _ := zip.NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	have := map[string]bool{}
	for _, f := range zr.File {
		have[f.Name] = true
	}
	for _, want := range []string{
		"span_labels/aggregated/Actor/metrics.csv",
		"document_labels/aggregated/Relevant/metrics.csv",
		"aggregated/confidence.csv",
	} {
		if !have[want] {
			t.Errorf("missing %s", want)
		}
	}
}
