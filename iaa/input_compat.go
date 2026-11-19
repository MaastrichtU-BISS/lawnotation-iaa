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

// annotationLevelOf reports the task as "span", "document" or "hybrid".
//
// The payload Lawnotation's server builds for this service already says
// which, under annotation_level. Its download export does not: there the
// field is span_granularity — how selections snap while annotating, "symbol",
// "word", "sentence", "paragraph" or "document" — and exports from before the
// rename call that same value annotation_level. Only "document" changes how a
// task is scored; any other granularity is a span task, or a hybrid one when
// it carries a second, whole-document labelset.
func annotationLevelOf(raw InputData) string {
	level := raw.AnnotationLevel
	if level == "" {
		level = raw.SpanGranularity
	}
	switch {
	case level == "document" || level == "hybrid":
		return level
	case len(raw.DocumentLabelset.Labels) > 0:
		return "hybrid"
	default:
		return "span"
	}
}
