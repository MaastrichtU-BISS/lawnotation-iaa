package iaa

import "encoding/json"

// Lawnotation's database column is still assignments.difficulty_rating, while
// its own exported vocabulary and this service's /metrics response both call
// the value confidence. Accepting the legacy name keeps ratings from silently
// decoding as 0 (unrated) when this service is deployed ahead of Lawnotation.
func (a *Assignment) UnmarshalJSON(data []byte) error {
	type assignment Assignment
	var raw struct {
		assignment
		LegacyRating *int `json:"difficulty_rating"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*a = Assignment(raw.assignment)
	if a.ConfidenceRating == 0 && raw.LegacyRating != nil {
		a.ConfidenceRating = *raw.LegacyRating
	}
	return nil
}

// annotationLevelOf reads the task's level under either name.
//
// Lawnotation's download export calls it span_granularity; the payload its
// server builds for this service calls it annotation_level. Reading only the
// latter meant a downloaded document-level export silently took the span
// path: spans there have no extent, so it would have matched zero-length
// spans and tokenized text that no annotation points into.
func annotationLevelOf(raw InputData) string {
	for _, level := range []string{raw.AnnotationLevel, raw.SpanGranularity} {
		if level != "" {
			return level
		}
	}
	return "span"
}
