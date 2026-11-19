// server.go — minimal HTTP backend exposing the IAA tool's functionality to
// a web client. No auth/CORS handling yet; both are expected to be layered
// on later.
package iaa

import (
	"archive/zip"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
)

// metricsResponse is the JSON body returned by POST /metrics.
type metricsResponse struct {
	AnnotationMetrics Report                  `json:"annotation_metrics"`
	ConfidenceMetrics ConfidenceRatingSummary `json:"confidence_metrics"`
}

// The HTTP API takes no scoring parameters. The span error taxonomy reports
// strict and lenient scoring from one matching pass, which is what the
// exact/contained criterion used to choose between, and coverage agreement is
// always word-level — character units multiply the matrix without telling the
// reader anything the word view does not.
const (
	apiCriterion   = "exact"
	apiGranularity = "word"
)

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// requireAPIKey wraps a handler with a check that the request carries
// "Authorization: Bearer <apiKey>". If apiKey is empty, auth is skipped
// entirely (used for local/dev runs where IAA_API_KEY isn't set). It guards
// the whole route set rather than each route, so a route added later cannot
// be left unprotected by forgetting to wrap it.
func requireAPIKey(next http.Handler, apiKey string) http.Handler {
	if apiKey == "" {
		return next
	}
	expected := "Bearer " + apiKey
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := r.Header.Get("Authorization")
		if subtle.ConstantTimeCompare([]byte(got), []byte(expected)) != 1 {
			writeJSONError(w, http.StatusUnauthorized, "unauthorized")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// handleMetrics handles POST /metrics.
// Body: the LawNotation task JSON. Returns annotation + confidence metrics
// as JSON, for display in a web client.
func handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	data, err := loadDataFromReader(r.Body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("invalid input JSON: %v", err))
		return
	}

	report := computeIAAFromData("", data, apiCriterion, apiGranularity)
	confidence := confidenceRatingSummary(data.Documents, data.Annotators)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(metricsResponse{
		AnnotationMetrics: report,
		ConfidenceMetrics: confidence,
	})
}

// handleReportZip handles POST /report.zip.
// Body: the LawNotation task JSON. Returns the same ZIP the CLI produces.
func handleReportZip(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	data, err := loadDataFromReader(r.Body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("invalid input JSON: %v", err))
		return
	}

	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", `attachment; filename="iaa_report.zip"`)
	zw := zip.NewWriter(w)
	if err := writeReportZip(zw, data.Documents, data.Labels, data.Annotators, apiCriterion, apiGranularity, data.AnnotationLevel, nil); err != nil {
		log.Printf("error writing report zip: %v", err)
		return
	}
	if err := zw.Close(); err != nil {
		log.Printf("error closing zip writer: %v", err)
	}
}

// Handler returns the tool's HTTP routes as a mountable handler, so a host
// application can serve them in-process instead of running this as a separate
// service. Paths are relative to wherever it is mounted:
//
//	POST metrics
//	POST report.zip
//
// Deliberately unauthenticated. The API key exists to protect a service
// exposed over a network; a host that has compiled this in is already inside
// the trust boundary, reaches it same-origin, and has no key to present.
// runServer is what adds auth, because that is the deployment that needs it.
func Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/metrics", handleMetrics)
	mux.HandleFunc("/report.zip", handleReportZip)
	return mux
}

// runServer starts the standalone HTTP server, requiring an API key when
// IAA_API_KEY is set. No CORS handling — this mode is for local testing or
// being placed behind a trusted proxy.
func runServer(port string) {
	apiKey := os.Getenv("IAA_API_KEY")

	addr := ":" + port
	log.Printf("IAA server listening on %s", addr)
	log.Printf("  POST /metrics     (body: LawNotation task JSON)")
	log.Printf("  POST /report.zip  (body: LawNotation task JSON)")
	if apiKey == "" {
		log.Printf("WARNING: IAA_API_KEY not set — endpoints are unauthenticated")
	} else {
		log.Printf("API key auth enabled (requests must send Authorization: Bearer <key>)")
	}
	if err := http.ListenAndServe(addr, requireAPIKey(Handler(), apiKey)); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
