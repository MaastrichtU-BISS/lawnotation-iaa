package iaa

import "testing"

func doc(text string, perAnnotator map[string][]Annotation) Document {
	d := Document{Name: "d", FullText: text}
	for annotator, anns := range perAnnotator {
		d.Assignments = append(d.Assignments, Assignment{Annotator: annotator, Annotations: anns})
	}
	return d
}

func ann(start, end int, label string) Annotation {
	return Annotation{Start: start, End: end, Label: label}
}

var twoLabels = map[string]bool{"A": true, "B": true}

func profileOf(t *testing.T, first, second []Annotation) SpanErrorProfile {
	t.Helper()
	d := doc("the quick brown fox jumps over the lazy dog", map[string][]Annotation{
		"1": first,
		"2": second,
	})
	profile, compared := spanErrorProfile([]Document{d}, twoLabels, "1", "2")
	if compared != 1 {
		t.Fatalf("expected 1 document compared, got %d", compared)
	}
	return profile
}

func TestClassifiesEachCategory(t *testing.T) {
	cases := []struct {
		name   string
		first  []Annotation
		second []Annotation
		want   SpanErrorProfile
	}{
		{
			name:   "identical span and label is a true positive",
			first:  []Annotation{ann(4, 9, "A")},
			second: []Annotation{ann(4, 9, "A")},
			want:   SpanErrorProfile{TruePositives: 1},
		},
		{
			name:   "same span, different label is a label error",
			first:  []Annotation{ann(4, 9, "A")},
			second: []Annotation{ann(4, 9, "B")},
			want:   SpanErrorProfile{LabelErrors: 1},
		},
		{
			name:   "second annotator marked less text",
			first:  []Annotation{ann(4, 15, "A")},
			second: []Annotation{ann(4, 9, "A")},
			want:   SpanErrorProfile{BoundarySmaller: 1},
		},
		{
			name:   "second annotator marked more text",
			first:  []Annotation{ann(4, 9, "A")},
			second: []Annotation{ann(4, 15, "A")},
			want:   SpanErrorProfile{BoundaryLarger: 1},
		},
		{
			name:   "overlap where neither contains the other",
			first:  []Annotation{ann(4, 12, "A")},
			second: []Annotation{ann(9, 19, "A")},
			want:   SpanErrorProfile{BoundaryOverlap: 1},
		},
		{
			name:   "different boundaries and different label",
			first:  []Annotation{ann(4, 12, "A")},
			second: []Annotation{ann(9, 19, "B")},
			want:   SpanErrorProfile{LabelAndBoundary: 1},
		},
		{
			name:   "no counterpart on either side",
			first:  []Annotation{ann(0, 3, "A")},
			second: []Annotation{ann(20, 25, "A")},
			want:   SpanErrorProfile{OnlyFirst: 1, OnlySecond: 1},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := profileOf(t, c.first, c.second); got != c.want {
				t.Errorf("got %+v, want %+v", got, c.want)
			}
		})
	}
}

// A label error is only visible when matching sees every label at once, which
// is what the per-label span matching cannot do.
func TestLabelErrorNeedsAllLabelsInOnePool(t *testing.T) {
	profile := profileOf(t, []Annotation{ann(4, 9, "A")}, []Annotation{ann(4, 9, "B")})
	if profile.LabelErrors != 1 {
		t.Fatalf("expected the disagreement to be a label error, got %+v", profile)
	}
	if profile.OnlyFirst != 0 || profile.OnlySecond != 0 {
		t.Errorf("a label error must not also count as unmatched spans: %+v", profile)
	}
}

func TestSwappingAnnotatorsMirrorsProfileAndKeepsF1(t *testing.T) {
	first := []Annotation{ann(4, 15, "A"), ann(20, 25, "A")}
	second := []Annotation{ann(4, 9, "A"), ann(30, 35, "B")}

	forward := profileOf(t, first, second)
	backward := profileOf(t, second, first)

	if forward.BoundarySmaller != backward.BoundaryLarger {
		t.Errorf("smaller/larger should mirror: %+v vs %+v", forward, backward)
	}
	if forward.OnlyFirst != backward.OnlySecond || forward.OnlySecond != backward.OnlyFirst {
		t.Errorf("only_first/only_second should mirror: %+v vs %+v", forward, backward)
	}
	if forward.strictF1() != backward.strictF1() {
		t.Errorf("strict F1 must be symmetric: %v vs %v", forward.strictF1(), backward.strictF1())
	}
	if forward.lenientF1() != backward.lenientF1() {
		t.Errorf("lenient F1 must be symmetric: %v vs %v", forward.lenientF1(), backward.lenientF1())
	}
}

// Strict scoring gives a partial match no credit: it counts as a spurious span
// and a missed one, exactly like a span only one annotator drew.
func TestStrictGivesPartialMatchesNoCredit(t *testing.T) {
	documents := []Document{doc("the quick brown fox jumps over the lazy dog",
		map[string][]Annotation{
			"1": {ann(4, 9, "A"), ann(10, 15, "A"), ann(20, 25, "A")},
			"2": {ann(4, 9, "A"), ann(10, 19, "A"), ann(30, 35, "A")},
		})}

	profile, _ := spanErrorProfile(documents, twoLabels, "1", "2")

	if profile.TruePositives != 1 || profile.BoundaryLarger != 1 ||
		profile.OnlyFirst != 1 || profile.OnlySecond != 1 {
		t.Fatalf("unexpected profile: %+v", profile)
	}
	// tp=1, fp = only_second + boundary = 2, fn = only_first + boundary = 2
	// -> 2/(2+2+2) = 0.3333
	if got := profile.strictF1(); !got.Valid || got.Value != 0.3333 {
		t.Errorf("strict F1 = %+v, want 0.3333", got)
	}
}

func TestLenientRewardsPartialMatchesAndGapShowsHowMuch(t *testing.T) {
	// One exact match and one boundary disagreement.
	profile := profileOf(t,
		[]Annotation{ann(4, 9, "A"), ann(10, 19, "A")},
		[]Annotation{ann(4, 9, "A"), ann(10, 15, "A")})

	strict, lenient := profile.strictF1(), profile.lenientF1()
	if !strict.Valid || !lenient.Valid {
		t.Fatal("expected both scores to be defined")
	}
	if lenient.Value <= strict.Value {
		t.Errorf("lenient %v should exceed strict %v when a boundary differs", lenient.Value, strict.Value)
	}

	// strict: tp=1, fp=fn=1  -> 2/(2+1+1) = 0.5
	// lenient: tp=1, fp=fn=0.5 -> 2/(2+0.5+0.5) = 0.6667
	if strict.Value != 0.5 {
		t.Errorf("strict F1 = %v, want 0.5", strict.Value)
	}
	if lenient.Value != 0.6667 {
		t.Errorf("lenient F1 = %v, want 0.6667", lenient.Value)
	}

	agreement := profile.agreement(1)
	if agreement.Gap.Value != roundFloat(lenient.Value-strict.Value).Value {
		t.Errorf("gap %v should be lenient minus strict", agreement.Gap.Value)
	}
}

// With no disagreements the two scorings must coincide, so the gap is the
// signal that partial matches are carrying the score.
func TestNoDisagreementsMeansNoGap(t *testing.T) {
	profile := profileOf(t,
		[]Annotation{ann(4, 9, "A"), ann(20, 25, "A")},
		[]Annotation{ann(4, 9, "A"), ann(30, 35, "A")})

	if profile.Disagreements() != 0 {
		t.Fatalf("expected no disagreements, got %+v", profile)
	}
	agreement := profile.agreement(1)
	if agreement.Gap.Value != 0 {
		t.Errorf("expected no gap, got %v", agreement.Gap.Value)
	}
}

func TestMatchingIsOneToOneAndDeterministic(t *testing.T) {
	// Two candidates overlap the same span; only one may claim it.
	profile := profileOf(t,
		[]Annotation{ann(4, 19, "A")},
		[]Annotation{ann(4, 9, "A"), ann(10, 19, "A")})

	if profile.total() != 2 {
		t.Errorf("one span against two should account for 2 items, got %+v", profile)
	}
	if profile.OnlySecond != 1 {
		t.Errorf("the losing candidate should be unmatched, got %+v", profile)
	}

	// Reversing the input order must not change the outcome.
	reversed := profileOf(t,
		[]Annotation{ann(4, 19, "A")},
		[]Annotation{ann(10, 19, "A"), ann(4, 9, "A")})
	if profile != reversed {
		t.Errorf("order changed the result: %+v vs %+v", profile, reversed)
	}
}

func TestUnknownLabelsAreIgnored(t *testing.T) {
	profile := profileOf(t,
		[]Annotation{ann(4, 9, "A"), ann(20, 25, "NotInLabelset")},
		[]Annotation{ann(4, 9, "A")})

	if profile != (SpanErrorProfile{TruePositives: 1}) {
		t.Errorf("a label outside the labelset should be dropped, got %+v", profile)
	}
}

func TestAnnotationLevelIsSpanDocumentOrHybrid(t *testing.T) {
	cases := []struct {
		name string
		raw  InputData
		want string
	}{
		{"annotation_level wins", InputData{AnnotationLevel: "document", SpanGranularity: "word"}, "document"},
		{"span_granularity is accepted", InputData{SpanGranularity: "document"}, "document"},
		{"neither given falls back to span", InputData{}, "span"},
		{"a granularity is a span task", InputData{SpanGranularity: "symbol"}, "span"},
		{"an old export's granularity is a span task", InputData{AnnotationLevel: "word"}, "span"},
		{"a document labelset makes it hybrid", InputData{
			SpanGranularity:  "word",
			DocumentLabelset: Labelset{Labels: []Label{{Name: "Relevant"}}},
		}, "hybrid"},
		{"hybrid is kept", InputData{AnnotationLevel: "hybrid"}, "hybrid"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := annotationLevelOf(c.raw); got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

// Two labels on identical offsets is common in this data, and it is exactly
// the case where every other tie-break is equal.
func TestSameOffsetsUnderTwoLabelsStillMatchesTheSameLabel(t *testing.T) {
	profile := profileOf(t,
		[]Annotation{ann(4, 9, "A")},
		[]Annotation{ann(4, 9, "B"), ann(4, 9, "A")})

	if profile.TruePositives != 1 {
		t.Errorf("the same-label span should claim the match, got %+v", profile)
	}
	if profile.OnlySecond != 1 {
		t.Errorf("the other label's span should be left unmatched, got %+v", profile)
	}
}

func TestResultDoesNotDependOnAnnotationOrder(t *testing.T) {
	first := []Annotation{ann(4, 9, "A"), ann(4, 9, "B"), ann(10, 19, "A"), ann(20, 25, "B")}
	second := []Annotation{ann(4, 9, "B"), ann(4, 9, "A"), ann(10, 15, "A"), ann(30, 35, "A")}

	want := profileOf(t, first, second)
	// Every rotation of both sides must classify identically.
	for shift := 1; shift < len(first); shift++ {
		rotate := func(anns []Annotation, by int) []Annotation {
			out := make([]Annotation, 0, len(anns))
			for i := range anns {
				out = append(out, anns[(i+by)%len(anns)])
			}
			return out
		}
		got := profileOf(t, rotate(first, shift), rotate(second, shift))
		if got != want {
			t.Errorf("rotating by %d changed the result:\n got %+v\nwant %+v", shift, got, want)
		}
	}
}
