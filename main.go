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
func metricsCSV(docs []Document, label string, annotators []string, criterion, granularity string) []byte {
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
	_ = w.Write([]string{"granularity", granularity})
	_ = w.Write([]string{"criterion", criterion})
	_ = w.Write([]string{"spans_per_annotator"})
	for _, annotator := range annotators {
		count := annCounts[annotator]
		_ = w.Write([]string{"annotator_" + annotator, fmt.Sprintf("%d", count)})
	}
	_ = w.Write([]string{""})

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
	pairs2 := spanMatchingAllPairs(docs, label, annotators, criterion)
	var f1vals []float64
	for _, pairResult := range pairs2 {
		for _, dir := range pairResult {
			if dir.F1.Valid {
				f1vals = append(f1vals, dir.F1.Value)
			}
		}
	}
	macroF1 := safeMean(f1vals)
	_ = w.Write([]string{"macro_f1", nullFloatStr(macroF1)})
	_ = w.Write([]string{""})

	// — Coverage Agreement —
	_ = w.Write([]string{"COVERAGE AGREEMENT"})
	covMatrix := buildCoverageMatrix(docs, label, annotators, granularity)
	_ = w.Write([]string{"matrix_items", fmt.Sprintf("%d", len(covMatrix[1]))})
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

	w.Flush()
	return []byte(buf.String())
}

// labelAnnotationsCSV generates annotations.csv content for one document+label.
func labelAnnotationsCSV(doc Document, label string) []byte {
	var buf strings.Builder
	w := csv.NewWriter(&buf)
	_ = w.Write([]string{"annotator", "start", "end", "text"})
	for _, asgn := range doc.Assignments {
		annotatorID := fmt.Sprintf("%v", asgn.Annotator)
		for _, ann := range asgn.Annotations {
			if ann.Label == label {
				_ = w.Write([]string{
					annotatorID,
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
// with an additional "document" column identifying the source document.
func aggregatedAnnotationsCSV(documents []Document, label string) []byte {
	var buf strings.Builder
	w := csv.NewWriter(&buf)
	_ = w.Write([]string{"document", "annotator", "start", "end", "text"})
	for _, doc := range documents {
		for _, asgn := range doc.Assignments {
			annotatorID := fmt.Sprintf("%v", asgn.Annotator)
			for _, ann := range asgn.Annotations {
				if ann.Label == label {
					_ = w.Write([]string{
						stripExt(doc.Name),
						annotatorID,
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

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

func main() {
	input := flag.String("input", "", "Path to LawNotation JSON export (required)")
	criterion := flag.String("criterion", "exact", "Match criterion: exact|contained")
	granularity := flag.String("granularity", "word", "Coverage granularity: char|word")
	output := flag.String("output", "iaa_report.zip", "Output ZIP path")
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

	// Load raw data (needed for per-document loop)
	labels, _, documents, err := loadData(*input)
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

	// Per-document CSVs
	fmt.Printf("Processing %d documents...\n", len(documents))
	for i, doc := range documents {
		printProgress(i+1, len(documents), doc.Name)

		docName := sanitizeName(stripExt(doc.Name))
		if docName == "" {
			docName = fmt.Sprintf("document_%d", i+1)
		}
		docAnns := docAnnotators(doc)

		for _, label := range labels {
			labelDir := fmt.Sprintf("documents/%s/%s", docName, sanitizeName(label))

			metricsEntry, err := zw.Create(labelDir + "/metrics.csv")
			if err != nil {
				log.Fatalf("error creating %s/metrics.csv: %v", labelDir, err)
			}
			if _, err := metricsEntry.Write(metricsCSV([]Document{doc}, label, docAnns, *criterion, *granularity)); err != nil {
				log.Fatalf("error writing metrics CSV: %v", err)
			}

			annotationsEntry, err := zw.Create(labelDir + "/annotations.csv")
			if err != nil {
				log.Fatalf("error creating %s/annotations.csv: %v", labelDir, err)
			}
			if _, err := annotationsEntry.Write(labelAnnotationsCSV(doc, label)); err != nil {
				log.Fatalf("error writing annotations CSV: %v", err)
			}
		}

		docDir := fmt.Sprintf("documents/%s", docName)
		confidenceEntry, err := zw.Create(docDir + "/confidence.csv")
		if err != nil {
			log.Fatalf("error creating %s/confidence.csv: %v", docDir, err)
		}
		if _, err := confidenceEntry.Write(confidenceCSV([]Document{doc}, docAnns)); err != nil {
			log.Fatalf("error writing confidence CSV: %v", err)
		}
	}
	fmt.Fprintln(os.Stderr) // end progress bar line

	// aggregated folder: location=root computed over all documents
	_, allAnnotators, _, _ := loadData(*input)
	for _, label := range labels {
		aggDir := fmt.Sprintf("aggregated/%s", sanitizeName(label))

		aggMetrics, err := zw.Create(aggDir + "/metrics.csv")
		if err != nil {
			log.Fatalf("error creating %s/metrics.csv: %v", aggDir, err)
		}
		if _, err := aggMetrics.Write(metricsCSV(documents, label, allAnnotators, *criterion, *granularity)); err != nil {
			log.Fatalf("error writing aggregated metrics CSV: %v", err)
		}

		aggAnnotations, err := zw.Create(aggDir + "/annotations.csv")
		if err != nil {
			log.Fatalf("error creating %s/annotations.csv: %v", aggDir, err)
		}
		if _, err := aggAnnotations.Write(aggregatedAnnotationsCSV(documents, label)); err != nil {
			log.Fatalf("error writing aggregated annotations CSV: %v", err)
		}
	}

	aggConfidence, err := zw.Create("aggregated/confidence.csv")
	if err != nil {
		log.Fatalf("error creating aggregated/confidence.csv: %v", err)
	}
	if _, err := aggConfidence.Write(confidenceCSV(documents, allAnnotators)); err != nil {
		log.Fatalf("error writing aggregated confidence CSV: %v", err)
	}

	fmt.Printf("\nReport written to: %s\n", *output)
	fmt.Printf("  aggregated/{label}/metrics.csv + annotations.csv, aggregated/confidence.csv\n")
	fmt.Printf("  documents/{doc}/{label}/metrics.csv + annotations.csv, documents/{doc}/confidence.csv  (%d documents × %d labels)\n\n",
		len(documents), len(labels))

	// Summary table
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
			f1Str = fmt.Sprintf("%.4f", f1Val.Value)
		}
		if covA.Valid {
			covStr = fmt.Sprintf("%.4f", covA.Value)
		}
		if meanKappa.Valid {
			kappaStr = fmt.Sprintf("%.4f", meanKappa.Value)
		}
		fmt.Printf("%-*s%s%s%s\n", colW, label, f1Str, covStr, kappaStr)
	}
}
