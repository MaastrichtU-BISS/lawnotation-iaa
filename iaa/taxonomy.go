package iaa

// Ortmann's (2022) span error taxonomy, which replaces the exact/contained
// criterion with one matching pass reported two ways.
//
// Matching here runs across every label at once, unlike precisionRecallF1,
// which filters to one label before matching. That filter makes a label error
// unobservable by construction: if both sides hold only spans of label L,
// no pair can ever disagree about the label. Seeing LE and LBE at all
// requires the whole annotation set in one pool.

import (
	"fmt"
	"sort"
)

type labeledSpan struct {
	start, end int
	label      string
}

// SpanErrorProfile counts how two annotators' spans relate, one category per
// span or matched pair, so the categories partition both annotators' spans.
//
// Boundary categories are named from the first annotator's point of view: in
// an "A_vs_B" pair, BoundarySmaller means B marked less text than A. The
// counts are otherwise symmetric — swapping A and B exchanges OnlyFirst with
// OnlySecond and BoundarySmaller with BoundaryLarger, and leaves F1 unchanged.
type SpanErrorProfile struct {
	TruePositives    int `json:"true_positives"`
	LabelErrors      int `json:"label_errors"`
	BoundarySmaller  int `json:"boundary_smaller"`
	BoundaryLarger   int `json:"boundary_larger"`
	BoundaryOverlap  int `json:"boundary_overlap"`
	LabelAndBoundary int `json:"label_and_boundary_errors"`
	OnlyFirst        int `json:"only_first"`
	OnlySecond       int `json:"only_second"`
}

func (p SpanErrorProfile) BoundaryErrors() int {
	return p.BoundarySmaller + p.BoundaryLarger + p.BoundaryOverlap
}

// Disagreements counts the pairs both annotators found but described
// differently — the ones strict and lenient scoring weigh differently.
func (p SpanErrorProfile) Disagreements() int {
	return p.LabelErrors + p.BoundaryErrors() + p.LabelAndBoundary
}

func (p SpanErrorProfile) total() int {
	return p.TruePositives + p.Disagreements() + p.OnlyFirst + p.OnlySecond
}

func (p SpanErrorProfile) add(other SpanErrorProfile) SpanErrorProfile {
	p.TruePositives += other.TruePositives
	p.LabelErrors += other.LabelErrors
	p.BoundarySmaller += other.BoundarySmaller
	p.BoundaryLarger += other.BoundaryLarger
	p.BoundaryOverlap += other.BoundaryOverlap
	p.LabelAndBoundary += other.LabelAndBoundary
	p.OnlyFirst += other.OnlyFirst
	p.OnlySecond += other.OnlySecond
	return p
}

// PairAgreement scores one annotator pair's profile both ways. The two F1s
// bracket the honest answer: strict allows no partial credit, lenient gives a
// disagreement half the weight of a span neither annotator's counterpart saw.
// Gap is how much of the score rests on partial matches.
type PairAgreement struct {
	Profile   SpanErrorProfile `json:"profile"`
	StrictF1  NullFloat64      `json:"strict_f1"`
	LenientF1 NullFloat64      `json:"lenient_f1"`
	Gap       NullFloat64      `json:"gap"`
	Documents int              `json:"documents_compared"`
}

// f1 is the harmonic mean written so it stays symmetric: swapping the two
// annotators exchanges fp and fn, which leaves 2tp/(2tp+fp+fn) untouched.
// That symmetry is why F1 survives the move from evaluation to agreement,
// while precision and recall separately do not — those need a gold standard.
func f1(tp, fp, fn float64) NullFloat64 {
	denom := 2*tp + fp + fn
	if denom == 0 {
		return noFloat()
	}
	return roundFloat(2 * tp / denom)
}

// strictF1 charges every disagreement a full false positive and a full false
// negative, reproducing what criterion="exact" scored.
func (p SpanErrorProfile) strictF1() NullFloat64 {
	d := float64(p.Disagreements())
	return f1(float64(p.TruePositives), float64(p.OnlySecond)+d, float64(p.OnlyFirst)+d)
}

// lenientF1 is Ortmann's "fair" scoring: one disagreement costs 0.5 FP + 0.5
// FN, since both annotators did find something there.
func (p SpanErrorProfile) lenientF1() NullFloat64 {
	d := float64(p.Disagreements()) / 2
	return f1(float64(p.TruePositives), float64(p.OnlySecond)+d, float64(p.OnlyFirst)+d)
}

func (p SpanErrorProfile) agreement(documents int) PairAgreement {
	strict, lenient := p.strictF1(), p.lenientF1()
	gap := noFloat()
	if strict.Valid && lenient.Valid {
		gap = roundFloat(lenient.Value - strict.Value)
	}
	return PairAgreement{
		Profile:   p,
		StrictF1:  strict,
		LenientF1: lenient,
		Gap:       gap,
		Documents: documents,
	}
}

func overlapLen(a, b labeledSpan) int {
	lo, hi := max(a.start, b.start), min(a.end, b.end)
	if hi <= lo {
		return 0
	}
	return hi - lo
}

func (s labeledSpan) length() int { return s.end - s.start }

// classify places one matched pair in its category, counted into the profile.
func (p *SpanErrorProfile) classify(a, b labeledSpan) {
	sameBounds := a.start == b.start && a.end == b.end
	sameLabel := a.label == b.label

	switch {
	case sameBounds && sameLabel:
		p.TruePositives++
	case sameBounds:
		p.LabelErrors++
	case !sameLabel:
		p.LabelAndBoundary++
	case b.start >= a.start && b.end <= a.end:
		p.BoundarySmaller++
	case a.start >= b.start && a.end <= b.end:
		p.BoundaryLarger++
	default:
		p.BoundaryOverlap++
	}
}

// matchLabeledSpans pairs spans 1:1, taking the best match first.
//
// Greedy rather than a global maximum-weight matching: annotations over real
// text overlap sparsely, so the two agree on all but contrived input, and that
// is worth more here than the last fraction of a percent.
//
// The ordering below decides every tie down to the spans' own indices, so the
// result depends only on the set of spans and not on the order they arrive in.
// That matters more than it sounds: annotators put two labels on the identical
// offsets often enough that, without it, reordering a file's annotations moved
// the task-wide score by several thousandths.
func matchLabeledSpans(first, second []labeledSpan) (pairs [][2]int, unmatchedFirst, unmatchedSecond []int) {
	type candidate struct {
		i, j, overlap, lengthDiff int
		exactBounds, sameLabel    bool
	}
	var candidates []candidate
	for i, a := range first {
		for j, b := range second {
			if n := overlapLen(a, b); n > 0 {
				diff := a.length() - b.length()
				if diff < 0 {
					diff = -diff
				}
				candidates = append(candidates, candidate{
					i: i, j: j, overlap: n, lengthDiff: diff,
					exactBounds: a.start == b.start && a.end == b.end,
					sameLabel:   a.label == b.label,
				})
			}
		}
	}
	// Best match first: most text in common, then agreeing on the boundaries,
	// then on the label, then closest in length, then leftmost. Boundaries and
	// label are in the chain because a span can sit at the same offsets under
	// two labels — common in this data — and without them such a pair ties on
	// every other key, so which one claimed the match came down to the order
	// the annotations happened to be stored in. The index comparisons last
	// leave no ties at all, since sort.Slice is not stable.
	sort.Slice(candidates, func(m, n int) bool {
		x, y := candidates[m], candidates[n]
		if x.overlap != y.overlap {
			return x.overlap > y.overlap
		}
		if x.exactBounds != y.exactBounds {
			return x.exactBounds
		}
		if x.sameLabel != y.sameLabel {
			return x.sameLabel
		}
		if x.lengthDiff != y.lengthDiff {
			return x.lengthDiff < y.lengthDiff
		}
		if first[x.i].start != first[y.i].start {
			return first[x.i].start < first[y.i].start
		}
		if first[x.i].end != first[y.i].end {
			return first[x.i].end < first[y.i].end
		}
		if second[x.j].start != second[y.j].start {
			return second[x.j].start < second[y.j].start
		}
		if second[x.j].end != second[y.j].end {
			return second[x.j].end < second[y.j].end
		}
		if first[x.i].label != first[y.i].label {
			return first[x.i].label < first[y.i].label
		}
		if second[x.j].label != second[y.j].label {
			return second[x.j].label < second[y.j].label
		}
		// Only reached when both spans are identical on both sides, where the
		// classification is the same whichever wins.
		if x.i != y.i {
			return x.i < y.i
		}
		return x.j < y.j
	})

	takenFirst := make([]bool, len(first))
	takenSecond := make([]bool, len(second))
	for _, c := range candidates {
		if takenFirst[c.i] || takenSecond[c.j] {
			continue
		}
		takenFirst[c.i], takenSecond[c.j] = true, true
		pairs = append(pairs, [2]int{c.i, c.j})
	}
	for i, taken := range takenFirst {
		if !taken {
			unmatchedFirst = append(unmatchedFirst, i)
		}
	}
	for j, taken := range takenSecond {
		if !taken {
			unmatchedSecond = append(unmatchedSecond, j)
		}
	}
	return pairs, unmatchedFirst, unmatchedSecond
}

// labeledSpansFor collects one annotator's spans. labels restricts the pool to
// the task's declared labels, so a stray label cannot invent a category.
func labeledSpansFor(anns []Annotation, labels map[string]bool) []labeledSpan {
	spans := make([]labeledSpan, 0, len(anns))
	for _, a := range anns {
		// A zero-extent annotation is a whole-document tag, not a span. It
		// can never overlap anything, so leaving it in would only add an
		// unmatched item to every comparison.
		if labels[a.Label] && a.End > a.Start {
			spans = append(spans, labeledSpan{a.Start, a.End, a.Label})
		}
	}
	return spans
}

// spanErrorProfile classifies every span of two annotators across all
// documents they both worked on.
func spanErrorProfile(documents []Document, labels map[string]bool, first, second string) (SpanErrorProfile, int) {
	var profile SpanErrorProfile
	compared := 0
	for _, doc := range documents {
		annMap := buildAnnMap(doc)
		firstAnns, hasFirst := annMap[first]
		secondAnns, hasSecond := annMap[second]
		if !hasFirst || !hasSecond {
			continue
		}
		compared++
		firstSpans := labeledSpansFor(firstAnns, labels)
		secondSpans := labeledSpansFor(secondAnns, labels)

		pairs, onlyFirst, onlySecond := matchLabeledSpans(firstSpans, secondSpans)
		for _, pair := range pairs {
			profile.classify(firstSpans[pair[0]], secondSpans[pair[1]])
		}
		profile.OnlyFirst += len(onlyFirst)
		profile.OnlySecond += len(onlySecond)
	}
	return profile, compared
}

// SpanAgreement is one entry per annotator pair, keyed "A_vs_B".
type SpanAgreement struct {
	Pairs map[string]PairAgreement `json:"pairs"`
}

// spanAgreement reports the taxonomy over the given labels. Only ever called
// with one label now, which is why the label-error categories always read
// zero — they need two labels in the same pool to occur at all. The
// classification stays complete so a cross-label view can be rebuilt without
// reworking the matcher.
func spanAgreement(documents []Document, labels, annotators []string) SpanAgreement {
	labelSet := make(map[string]bool, len(labels))
	for _, l := range labels {
		labelSet[l] = true
	}
	pairs := map[string]PairAgreement{}
	for i := 0; i < len(annotators); i++ {
		for j := i + 1; j < len(annotators); j++ {
			a, b := annotators[i], annotators[j]
			profile, compared := spanErrorProfile(documents, labelSet, a, b)
			pairs[fmt.Sprintf("%s_vs_%s", a, b)] = profile.agreement(compared)
		}
	}
	return SpanAgreement{Pairs: pairs}
}

// markedColumnCount counts the units at least one annotator marked. Units
// where every assigned annotator scored 0 carry no information about the
// label: they are agreed silence, and over a corpus they outnumber
// everything else.
func markedColumnCount(matrix [][]*float64) int {
	if len(matrix) == 0 {
		return 0
	}
	count := 0
	for col := 0; col < len(matrix[0]); col++ {
		for row := range matrix {
			if v := matrix[row][col]; v != nil && *v > 0 {
				count++
				break
			}
		}
	}
	return count
}
