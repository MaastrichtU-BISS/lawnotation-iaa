// iaa.go — CLI entry point and output formatting for the IAA tool.
// All computation logic (types, metrics, data loading) lives in iaa.go.
package main

import (
	"archive/zip"
	"encoding/csv"
	"encoding/json"
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

// labelCSVBytes generates CSV content for one document+label.
func labelCSVBytes(doc Document, label string, annotators []string, criterion, granularity string) []byte {
	docs := []Document{doc}
	var buf strings.Builder
	w := csv.NewWriter(&buf)

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

	_ = w.Write([]string{""})

	// — Coverage Agreement —
	_ = w.Write([]string{"COVERAGE AGREEMENT"})
	_ = w.Write([]string{"metric", "value"})

	covMatrix := buildCoverageMatrix(docs, label, annotators, granularity)
	covAlpha := krippendorffAlpha(covMatrix)
	covKappas := cohenKappaAllPairs(covMatrix, annotators)

	_ = w.Write([]string{"krippendorff_alpha", nullFloatStr(covAlpha)})

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

	// Write aggregate JSON into zip
	aggEntry, err := zw.Create("aggregate.json")
	if err != nil {
		log.Fatalf("error creating aggregate.json in zip: %v", err)
	}
	enc := json.NewEncoder(aggEntry)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		log.Fatalf("error encoding aggregate JSON: %v", err)
	}

	// Per-document CSVs
	fmt.Printf("Processing %d documents...\n", len(documents))
	for i, doc := range documents {
		printProgress(i+1, len(documents), doc.Name)

		docName := sanitizeName(doc.Name)
		if docName == "" {
			docName = fmt.Sprintf("document_%d", i+1)
		}
		docAnns := docAnnotators(doc)

		for _, label := range labels {
			entryPath := fmt.Sprintf("documents/%s/%s.csv", docName, sanitizeName(label))
			csvEntry, err := zw.Create(entryPath)
			if err != nil {
				log.Fatalf("error creating zip entry %s: %v", entryPath, err)
			}
			data := labelCSVBytes(doc, label, docAnns, *criterion, *granularity)
			if _, err := csvEntry.Write(data); err != nil {
				log.Fatalf("error writing CSV entry: %v", err)
			}
		}
	}
	fmt.Fprintln(os.Stderr) // end progress bar line

	fmt.Printf("\nReport written to: %s\n", *output)
	fmt.Printf("  aggregate.json\n")
	fmt.Printf("  documents/{name}/{label}.csv  (%d documents × %d labels)\n\n",
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
