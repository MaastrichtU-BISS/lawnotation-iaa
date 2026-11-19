// cli.go — the command-line entry point. The report itself is written by
// export.go, and all computation lives in iaa.go and its siblings.
package iaa

import (
	"archive/zip"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

// loadDotEnv reads KEY=VALUE pairs from a .env file in the working
// directory, if one exists, and applies them via os.Setenv for any key not
// already present in the environment. Real environment variables always
// win over .env file contents. Silently does nothing if no .env is found.
func loadDotEnv() {
	data, err := os.ReadFile(".env")
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		if _, exists := os.LookupEnv(key); !exists {
			os.Setenv(key, value)
		}
	}
}

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
// Entry point
// ---------------------------------------------------------------------------

// RunCLI runs the command-line entry point: flag parsing, batch report
// generation and the summary table. Kept here rather than in the root main
// package so the whole tool stays one package, importable as a library.
func RunCLI() {
	loadDotEnv()

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

	data, err := loadData(*input)
	if err != nil {
		log.Fatalf("error loading data: %v", err)
	}
	fmt.Println("Computing agreement...")
	report := computeIAAFromData(*input, data, *criterion, *granularity)
	confidence := confidenceRatingSummary(data.Documents, data.Annotators)

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

	fmt.Printf("Comparing %d documents...\n", len(data.Documents))
	if err := writeReportZip(zw, data, report, confidence, printProgress); err != nil {
		log.Fatalf("error writing report: %v", err)
	}
	fmt.Fprintln(os.Stderr) // end progress bar line
	fmt.Printf("\nReport written to: %s (see README.md inside for what each table holds)\n\n", *output)

	printSummary(data, report)
}

// printSummary is the at-a-glance table: one line per label, worst agreement
// first being left to whoever reads it.
func printSummary(data TaskData, report Report) {
	fmt.Println("=== SUMMARY ===")
	fmt.Printf("  Documents        : %d\n", len(data.Documents))
	fmt.Printf("  Annotators       : %s\n", strings.Join(data.Annotators, ", "))
	fmt.Printf("  Annotation level : %s\n\n", report.Meta.AnnotationLevel)

	const labelWidth = 34
	fmt.Printf("%-*s %-9s %8s %10s %10s\n", labelWidth, "Label", "Kind", "alpha", "F1 strict", "F1 lenient")
	fmt.Println(strings.Repeat("-", labelWidth+41))
	cell := func(v NullFloat64) string {
		if !v.Valid {
			return "-"
		}
		return num(v)
	}
	for _, g := range labelGroups(data, report) {
		for _, label := range g.labels {
			r := g.results[label]
			strict, lenient := meanF1(r.SpanAgreement)
			fmt.Printf("%-*s %-9s %8s %10s %10s\n", labelWidth, label, g.kind,
				cell(r.CoverageAgreement.KrippendorffAlpha), cell(strict), cell(lenient))
		}
	}
}
