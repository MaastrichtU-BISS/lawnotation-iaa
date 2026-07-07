// iaa.go — CLI entry point and output formatting for the IAA tool.
// All computation logic (types, metrics, data loading) lives in iaa.go.
package main

import (
	"archive/zip"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

func printProgress(current, total int, docName string) {
	const width = 30
	pct := float64(current) / float64(total)
	filled := int(pct * float64(width))
	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	label := docName
	if len(label) > 40 {
		label = label[:37] + "..."
	}
	fmt.Fprintf(os.Stderr, "\r[%s] %d/%d (%.1f%%) %s   ", bar, current, total, pct*100, label)
}

// ---------------------------------------------------------------------------
// Per-document CSV helpers
// ---------------------------------------------------------------------------

func nullFloatStr(v NullFloat64) string {
	if !v.Valid {
		return "N/A"
	}
	return fmt.Sprintf("%.4f", math.Round(v.Value*10000)/10000)
}

// docAnnotators returns sorted annotator IDs present in a single document.
func docAnnotators(doc Document) []string {
	seen := map[string]bool{}
	for _, asgn := range doc.Assignments {
		seen[fmt.Sprintf("%v", asgn.Annotator)] = true
	}
	result := make([]string, 0, len(seen))
	for a := range seen {
		result = append(result, a)
	}
	sort.Strings(result)
	return result
}

// stripExt removes the file extension from a name (e.g. "doc.txt" → "doc").
func stripExt(name string) string {
	if ext := filepath.Ext(name); ext != "" {
		return name[:len(name)-len(ext)]
	}
	return name
}

// metricsCSV generates metrics.csv content for a set of documents and a single label.
// For annotationLevel=="document" data (annotations with no real span extent),
// Span Matching is not applicable and is omitted; Coverage Agreement becomes a
// document-level label-presence agreement instead of token coverage.
func metricsCSV(docs []Document, label string, annotators []string, criterion, granularity, annotationLevel string) []byte {
	isDocumentLevel := annotationLevel == "document"

	var buf strings.Builder
	w := csv.NewWriter(&buf)

	_ = w.Write([]string{"GENERAL"})
	annCounts := map[string]int{}
	for _, doc := range docs {
		for _, asgn := range doc.Assignments {
			annotatorID := fmt.Sprintf("%v", asgn.Annotator)
			for _, ann := range asgn.Annotations {
				if ann.Label == label {
					annCounts[annotatorID]++
				}
			}
		}
	}
	_ = w.Write([]string{"label", label})
	if len(docs) == 1 {
		_ = w.Write([]string{"document", stripExt(docs[0].Name)})
	} else {
		_ = w.Write([]string{"num_documents", fmt.Sprintf("%d", len(docs))})
	}
	_ = w.Write([]string{"annotation_level", annotationLevel})
	if isDocumentLevel {
		_ = w.Write([]string{"granularity", "N/A"})
		_ = w.Write([]string{"criterion", "N/A"})
	} else {
		_ = w.Write([]string{"granularity", granularity})
		_ = w.Write([]string{"criterion", criterion})
	}
	_ = w.Write([]string{"annotations_per_annotator"})
	for _, annotator := range annotators {
		count := annCounts[annotator]
		_ = w.Write([]string{annotator, fmt.Sprintf("%d", count)})
	}
	_ = w.Write([]string{""})

	if !isDocumentLevel {
		// — Span Matching —
		_ = w.Write([]string{"SPAN MATCHING"})
		_ = w.Write([]string{"pair", "direction", "tp", "ref_count", "sys_count", "precision", "recall", "f1"})
		pairs := spanMatchingAllPairs(docs, label, annotators, criterion)
		pairKeys := make([]string, 0, len(pairs))
		for k := range pairs {
			pairKeys = append(pairKeys, k)
		}
		sort.Strings(pairKeys)
		for _, pairKey := range pairKeys {
			dirMap := pairs[pairKey]
			dirKeys := make([]string, 0, len(dirMap))
			for k := range dirMap {
				dirKeys = append(dirKeys, k)
			}
			sort.Strings(dirKeys)
			for _, dirKey := range dirKeys {
				r := dirMap[dirKey]
				_ = w.Write([]string{
					pairKey, dirKey,
					fmt.Sprintf("%d", r.TruePositives),
					fmt.Sprintf("%d", r.RefSpanCount),
					fmt.Sprintf("%d", r.SysSpanCount),
					nullFloatStr(r.Precision),
					nullFloatStr(r.Recall),
					nullFloatStr(r.F1),
				})
			}
		}

		// macro F1 (mean across all pair directions)
		var f1vals []float64
		for _, pairResult := range pairs {
			for _, dir := range pairResult {
				if dir.F1.Valid {
					f1vals = append(f1vals, dir.F1.Value)
				}
			}
		}
		macroF1 := safeMean(f1vals)
		_ = w.Write([]string{"macro_f1", nullFloatStr(macroF1)})
		_ = w.Write([]string{""})
	}

	// — Coverage / Label Agreement —
	var covMatrix [][]*float64
	if isDocumentLevel {
		_ = w.Write([]string{"LABEL AGREEMENT"})
		covMatrix = buildPresenceMatrix(docs, label, annotators)
	} else {
		_ = w.Write([]string{"COVERAGE AGREEMENT"})
		covMatrix = buildCoverageMatrix(docs, label, annotators, granularity)
	}
	matrixItems := 0
	if len(covMatrix) > 0 {
		matrixItems = len(covMatrix[0])
	}
	_ = w.Write([]string{"matrix_items", fmt.Sprintf("%d", matrixItems)})
	covAlpha := krippendorffAlpha(covMatrix)
	covKappas := cohenKappaAllPairs(covMatrix, annotators)

	_ = w.Write([]string{"krippendorff_alpha", nullFloatStr(covAlpha)})
	_ = w.Write([]string{"cohen_kappa_pairs"})
	kappaKeys := make([]string, 0, len(covKappas))
	for k := range covKappas {
		kappaKeys = append(kappaKeys, k)
	}
	sort.Strings(kappaKeys)
	for _, k := range kappaKeys {
		_ = w.Write([]string{k, nullFloatStr(covKappas[k])})
	}

	w.Flush()
	return []byte(buf.String())
}

// confidenceCSV generates confidence.csv content: difficulty_rating stats
// and Krippendorff's alpha across annotators for a set of documents. This is
// independent of label, so it's written once per document (or once for the
// aggregate) rather than duplicated inside every label's metrics.csv.
func confidenceCSV(docs []Document, annotators []string) []byte {
	var buf strings.Builder
	w := csv.NewWriter(&buf)

	diff := difficultyRatingSummary(docs, annotators)
	_ = w.Write([]string{"total", fmt.Sprintf("%d", diff.Total)})
	_ = w.Write([]string{"rated", fmt.Sprintf("%d", diff.Rated)})
	_ = w.Write([]string{"mean", nullFloatStr(diff.Mean)})
	for star := 1; star <= 5; star++ {
		_ = w.Write([]string{fmt.Sprintf("star_%d", star), fmt.Sprintf("%d", diff.Counts[star])})
	}
	_ = w.Write([]string{"krippendorff_alpha", nullFloatStr(diff.Alpha)})
	_ = w.Write([]string{"krippendorff_alpha_pairs"})
	pairKeys := make([]string, 0, len(diff.AlphaPairs))
	for k := range diff.AlphaPairs {
		pairKeys = append(pairKeys, k)
	}
	sort.Strings(pairKeys)
	for _, k := range pairKeys {
		_ = w.Write([]string{k, nullFloatStr(diff.AlphaPairs[k])})
	}

	w.Flush()
	return []byte(buf.String())
}

// labelAnnotationsCSV generates annotations.csv content for one document+label.
// For annotationLevel=="document" data, start/end/text/difficulty_rating are
// omitted: start/end/text are always trivial (0, 0, "") at that level, and
// difficulty_rating is already reported per document/annotator in
// confidence.csv, so repeating it per annotation row here adds nothing.
func labelAnnotationsCSV(doc Document, label, annotationLevel string) []byte {
	var buf strings.Builder
	w := csv.NewWriter(&buf)
	if annotationLevel == "document" {
		_ = w.Write([]string{"annotator"})
		for _, asgn := range doc.Assignments {
			annotatorID := fmt.Sprintf("%v", asgn.Annotator)
			for _, ann := range asgn.Annotations {
				if ann.Label == label {
					_ = w.Write([]string{annotatorID})
				}
			}
		}
		w.Flush()
		return []byte(buf.String())
	}
	_ = w.Write([]string{"annotator", "difficulty_rating", "start", "end", "text"})
	for _, asgn := range doc.Assignments {
		annotatorID := fmt.Sprintf("%v", asgn.Annotator)
		for _, ann := range asgn.Annotations {
			if ann.Label == label {
				_ = w.Write([]string{
					annotatorID,
					fmt.Sprintf("%d", asgn.DifficultyRating),
					fmt.Sprintf("%d", ann.Start),
					fmt.Sprintf("%d", ann.End),
					ann.Text,
				})
			}
		}
	}
	w.Flush()
	return []byte(buf.String())
}

// aggregatedAnnotationsCSV generates annotations.csv for a label across all documents,
// with an additional "document" column identifying the source document. See
// labelAnnotationsCSV for why start/end/text/difficulty_rating are omitted
// for annotationLevel=="document" data.
func aggregatedAnnotationsCSV(documents []Document, label, annotationLevel string) []byte {
	var buf strings.Builder
	w := csv.NewWriter(&buf)
	if annotationLevel == "document" {
		_ = w.Write([]string{"document", "annotator"})
		for _, doc := range documents {
			for _, asgn := range doc.Assignments {
				annotatorID := fmt.Sprintf("%v", asgn.Annotator)
				for _, ann := range asgn.Annotations {
					if ann.Label == label {
						_ = w.Write([]string{stripExt(doc.Name), annotatorID})
					}
				}
			}
		}
		w.Flush()
		return []byte(buf.String())
	}
	_ = w.Write([]string{"document", "annotator", "difficulty_rating", "start", "end", "text"})
	for _, doc := range documents {
		for _, asgn := range doc.Assignments {
			annotatorID := fmt.Sprintf("%v", asgn.Annotator)
			for _, ann := range asgn.Annotations {
				if ann.Label == label {
					_ = w.Write([]string{
						stripExt(doc.Name),
						annotatorID,
						fmt.Sprintf("%d", asgn.DifficultyRating),
						fmt.Sprintf("%d", ann.Start),
						fmt.Sprintf("%d", ann.End),
						ann.Text,
					})
				}
			}
		}
	}
	w.Flush()
	return []byte(buf.String())
}

// sanitizeName replaces characters unsafe in zip entry paths.
func sanitizeName(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch r {
		case '/', '\\', ':', '*', '?', '"', '<', '>', '|':
			b.WriteRune('_')
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// writeReportZip writes the full report (per-document and aggregated
// metrics/annotations/confidence CSVs) into zw. onProgress, if non-nil, is
// called before each document is processed (used by the CLI for a progress
// bar; the server passes nil). Shared by both the CLI and the HTTP server so
// the zip contents can't drift between the two entry points.
func writeReportZip(zw *zip.Writer, documents []Document, labels, allAnnotators []string, criterion, granularity, annotationLevel string, onProgress func(current, total int, docName string)) error {
	for i, doc := range documents {
		if onProgress != nil {
			onProgress(i+1, len(documents), doc.Name)
		}

		docName := sanitizeName(stripExt(doc.Name))
		if docName == "" {
			docName = fmt.Sprintf("document_%d", i+1)
		}
		docAnns := docAnnotators(doc)

		for _, label := range labels {
			labelDir := fmt.Sprintf("documents/%s/%s", docName, sanitizeName(label))

			metricsEntry, err := zw.Create(labelDir + "/metrics.csv")
			if err != nil {
				return fmt.Errorf("creating %s/metrics.csv: %w", labelDir, err)
			}
			if _, err := metricsEntry.Write(metricsCSV([]Document{doc}, label, docAnns, criterion, granularity, annotationLevel)); err != nil {
				return fmt.Errorf("writing %s/metrics.csv: %w", labelDir, err)
			}

			annotationsEntry, err := zw.Create(labelDir + "/annotations.csv")
			if err != nil {
				return fmt.Errorf("creating %s/annotations.csv: %w", labelDir, err)
			}
			if _, err := annotationsEntry.Write(labelAnnotationsCSV(doc, label, annotationLevel)); err != nil {
				return fmt.Errorf("writing %s/annotations.csv: %w", labelDir, err)
			}
		}

		docDir := fmt.Sprintf("documents/%s", docName)
		confidenceEntry, err := zw.Create(docDir + "/confidence.csv")
		if err != nil {
			return fmt.Errorf("creating %s/confidence.csv: %w", docDir, err)
		}
		if _, err := confidenceEntry.Write(confidenceCSV([]Document{doc}, docAnns)); err != nil {
			return fmt.Errorf("writing %s/confidence.csv: %w", docDir, err)
		}
	}

	for _, label := range labels {
		aggDir := fmt.Sprintf("aggregated/%s", sanitizeName(label))

		aggMetrics, err := zw.Create(aggDir + "/metrics.csv")
		if err != nil {
			return fmt.Errorf("creating %s/metrics.csv: %w", aggDir, err)
		}
		if _, err := aggMetrics.Write(metricsCSV(documents, label, allAnnotators, criterion, granularity, annotationLevel)); err != nil {
			return fmt.Errorf("writing %s/metrics.csv: %w", aggDir, err)
		}

		aggAnnotations, err := zw.Create(aggDir + "/annotations.csv")
		if err != nil {
			return fmt.Errorf("creating %s/annotations.csv: %w", aggDir, err)
		}
		if _, err := aggAnnotations.Write(aggregatedAnnotationsCSV(documents, label, annotationLevel)); err != nil {
			return fmt.Errorf("writing %s/annotations.csv: %w", aggDir, err)
		}
	}

	aggConfidence, err := zw.Create("aggregated/confidence.csv")
	if err != nil {
		return fmt.Errorf("creating aggregated/confidence.csv: %w", err)
	}
	if _, err := aggConfidence.Write(confidenceCSV(documents, allAnnotators)); err != nil {
		return fmt.Errorf("writing aggregated/confidence.csv: %w", err)
	}

	return nil
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

func main() {
	input := flag.String("input", "", "Path to LawNotation JSON export (required unless --serve)")
	criterion := flag.String("criterion", "exact", "Match criterion: exact|contained")
	granularity := flag.String("granularity", "word", "Coverage granularity: char|word")
	output := flag.String("output", "iaa_report.zip", "Output ZIP path")
	serve := flag.Bool("serve", false, "Start an HTTP server (POST /metrics, POST /report.zip) instead of running as a one-off CLI batch")
	port := flag.String("port", "8080", "Port to listen on in --serve mode")
	flag.Parse()

	if *serve {
		runServer(*port)
		return
	}

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

	// Load raw data (needed for per-document loop)
	labels, allAnnotators, documents, annotationLevel, err := loadData(*input)
	if err != nil {
		log.Fatalf("error loading data: %v", err)
	}

	// Aggregate report
	fmt.Println("Computing aggregate report...")
	report, err := computeIAA(*input, *criterion, *granularity)
	if err != nil {
		log.Fatalf("error: %v", err)
	}

	// Create zip (ensure parent directory exists)
	if dir := filepath.Dir(*output); dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			log.Fatalf("error creating output directory: %v", err)
		}
	}
	zipFile, err := os.Create(*output)
	if err != nil {
		log.Fatalf("error creating zip: %v", err)
	}
	defer zipFile.Close()
	zw := zip.NewWriter(zipFile)
	defer zw.Close()

	fmt.Printf("Processing %d documents...\n", len(documents))
	if err := writeReportZip(zw, documents, labels, allAnnotators, *criterion, *granularity, annotationLevel, printProgress); err != nil {
		log.Fatalf("error writing report: %v", err)
	}
	fmt.Fprintln(os.Stderr) // end progress bar line

	fmt.Printf("\nReport written to: %s\n", *output)
	fmt.Printf("  aggregated/{label}/metrics.csv + annotations.csv, aggregated/confidence.csv\n")
	fmt.Printf("  documents/{doc}/{label}/metrics.csv + annotations.csv, documents/{doc}/confidence.csv  (%d documents × %d labels)\n\n",
		len(documents), len(labels))

	// Summary table
	isDocumentLevel := report.Meta.AnnotationLevel == "document"
	fmt.Println("=== SUMMARY ===")
	fmt.Printf("  Documents                          : %d\n", report.Meta.NumDocuments)
	fmt.Printf("  Annotators                         : %s\n", strings.Join(report.Meta.Annotators, ", "))
	fmt.Printf("  Annotation level                   : %s\n", report.Meta.AnnotationLevel)
	if !isDocumentLevel {
		fmt.Printf("  Criterion                      : %s\n", report.Meta.Criterion)
		fmt.Printf("  Granularity (coverage view)        : %s\n", report.Meta.Granularity)
	}
	fmt.Println()

	colW := 34
	if isDocumentLevel {
		fmt.Printf("%-*s %8s\n", colW, "Label", "label α")
		fmt.Println(strings.Repeat("-", 43))
	} else {
		fmt.Printf("%-*s %8s %8s\n", colW, "Label", "F1", "cov α")
		fmt.Println(strings.Repeat("-", 56))
	}

	labelNames := make([]string, 0, len(report.PerLabel))
	for k := range report.PerLabel {
		labelNames = append(labelNames, k)
	}
	sort.Strings(labelNames)

	for _, label := range labelNames {
		v := report.PerLabel[label]
		f1Val := v.SpanMatching.MacroF1
		covA := v.CoverageAgreement.KrippendorffAlpha

		f1Str, covStr := "     N/A", "     N/A"
		if f1Val.Valid {
			f1Str = fmt.Sprintf("%.4f", f1Val.Value)
		}
		if covA.Valid {
			covStr = fmt.Sprintf("%.4f", covA.Value)
		}
		if isDocumentLevel {
			fmt.Printf("%-*s%s\n", colW, label, covStr)
		} else {
			fmt.Printf("%-*s%s%s\n", colW, label, f1Str, covStr)
		}
	}
}
