// server.go — minimal HTTP backend exposing the IAA tool's functionality to
// a web client. No auth/CORS handling yet; both are expected to be layered
// on later.
package main

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
)

// metricsResponse is the JSON body returned by POST /metrics.
type metricsResponse struct {
	AnnotationMetrics Report                  `json:"annotation_metrics"`
	ConfidenceMetrics DifficultyRatingSummary `json:"confidence_metrics"`
}

// parseLevelParams reads criterion/granularity query params, applying the
// same defaults and validation as the CLI flags.
func parseLevelParams(r *http.Request) (criterion, granularity string, err error) {
	criterion = r.URL.Query().Get("criterion")
	if criterion == "" {
		criterion = "exact"
	}
	if criterion != "exact" && criterion != "contained" {
		return "", "", fmt.Errorf("criterion must be 'exact' or 'contained'")
	}
	granularity = r.URL.Query().Get("granularity")
	if granularity == "" {
		granularity = "word"
	}
	if granularity != "char" && granularity != "word" {
		return "", "", fmt.Errorf("granularity must be 'char' or 'word'")
	}
	return criterion, granularity, nil
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// handleMetrics handles POST /metrics?criterion=exact&granularity=word.
// Body: the LawNotation task JSON. Returns annotation + confidence metrics
// as JSON, for display in a web client.
func handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	criterion, granularity, err := parseLevelParams(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	labels, annotators, documents, annotationLevel, err := loadDataFromReader(r.Body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("invalid input JSON: %v", err))
		return
	}

	report := computeIAAFromData("", labels, annotators, documents, annotationLevel, criterion, granularity)
	confidence := difficultyRatingSummary(documents, annotators)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(metricsResponse{
		AnnotationMetrics: report,
		ConfidenceMetrics: confidence,
	})
}

// handleReportZip handles POST /report.zip?criterion=exact&granularity=word.
// Body: the LawNotation task JSON. Returns the same ZIP the CLI produces.
func handleReportZip(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	criterion, granularity, err := parseLevelParams(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	labels, annotators, documents, annotationLevel, err := loadDataFromReader(r.Body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("invalid input JSON: %v", err))
		return
	}

	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", `attachment; filename="iaa_report.zip"`)
	zw := zip.NewWriter(w)
	if err := writeReportZip(zw, documents, labels, annotators, criterion, granularity, annotationLevel, nil); err != nil {
		log.Printf("error writing report zip: %v", err)
		return
	}
	if err := zw.Close(); err != nil {
		log.Printf("error closing zip writer: %v", err)
	}
}

// runServer starts the HTTP server. No auth or CORS handling — this is for
// local testing only; both are expected to be added before any real
// deployment.
func runServer(port string) {
	mux := http.NewServeMux()
	mux.HandleFunc("/metrics", handleMetrics)
	mux.HandleFunc("/report.zip", handleReportZip)

	addr := ":" + port
	log.Printf("IAA server listening on %s", addr)
	log.Printf("  POST /metrics?criterion=exact&granularity=word     (body: LawNotation task JSON)")
	log.Printf("  POST /report.zip?criterion=exact&granularity=word  (body: LawNotation task JSON)")
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
