package iaa

import (
	"math"
	"testing"
)

// Krippendorff's alpha and Cohen's kappa answer the same question about the
// same two rows of a coverage matrix, differing only in how they model
// chance: kappa gives each annotator their own marking rate, alpha pools the
// two into one. These tests record how far apart that puts them, because the
// answer decides whether reporting both tells a reader anything.

// matrix builds a two-row coverage matrix from 0/1 patterns.
func matrix(a, b []float64) [][]*float64 {
	rows := make([][]*float64, 2)
	for i, vals := range [][]float64{a, b} {
		row := make([]*float64, len(vals))
		for j := range vals {
			v := vals[j]
			row[j] = &v
		}
		rows[i] = row
	}
	return rows
}

// coverage spreads `marked` ones over `n` positions, starting at `offset`, the
// way a coverage matrix looks when an annotator marks a contiguous stretch.
func coverage(n, marked, offset int) []float64 {
	out := make([]float64, n)
	for i := 0; i < marked; i++ {
		out[(offset+i)%n] = 1
	}
	return out
}

func bothCoefficients(t *testing.T, m [][]*float64) (alpha, kappa float64) {
	t.Helper()
	a := krippendorffAlpha(m)
	k := cohenKappaPair(m[0], m[1])
	if !a.Valid || !k.Valid {
		t.Fatalf("expected both coefficients to be defined, got alpha=%v kappa=%v", a, k)
	}
	return a.Value, k.Value
}

func TestAlphaAndKappaAgreeOnLargeBinaryMatrices(t *testing.T) {
	// The shape of real coverage data: thousands of positions, a small marked
	// fraction, the two annotators overlapping on most of it.
	const n = 40000
	m := matrix(coverage(n, 6000, 0), coverage(n, 6000, 500))

	alpha, kappa := bothCoefficients(t, m)
	if diff := math.Abs(alpha - kappa); diff > 0.005 {
		t.Errorf("alpha %.4f and kappa %.4f differ by %.4f; they should track each other "+
			"closely on binary data at this scale", alpha, kappa, diff)
	}
}

// The chance models only come apart when the two annotators mark at very
// different rates — kappa credits each one its own rate, alpha assumes they
// are drawn from the same distribution.
func TestAlphaAndKappaDivergeWhenMarkingRatesDiffer(t *testing.T) {
	const n = 40000
	lopsided := matrix(coverage(n, 20000, 0), coverage(n, 2000, 0))
	even := matrix(coverage(n, 11000, 0), coverage(n, 11000, 9000))

	lopsidedAlpha, lopsidedKappa := bothCoefficients(t, lopsided)
	evenAlpha, evenKappa := bothCoefficients(t, even)

	lopsidedGap := math.Abs(lopsidedAlpha - lopsidedKappa)
	evenGap := math.Abs(evenAlpha - evenKappa)

	t.Logf("lopsided rates (50%% vs 5%%): alpha %.4f kappa %.4f gap %.4f",
		lopsidedAlpha, lopsidedKappa, lopsidedGap)
	t.Logf("even rates (both 27.5%%):    alpha %.4f kappa %.4f gap %.4f",
		evenAlpha, evenKappa, evenGap)

	if lopsidedGap <= evenGap {
		t.Errorf("expected the gap to widen when marking rates differ: lopsided %.4f, even %.4f",
			lopsidedGap, evenGap)
	}
}

// The other source of difference is alpha's small-sample correction, which
// divides by n(n-1) where kappa divides by n^2. It vanishes as the matrix
// grows, so it cannot matter on a corpus.
func TestAlphaSmallSampleCorrectionFadesWithSize(t *testing.T) {
	small := matrix([]float64{1, 1, 0, 0, 1, 0, 0, 0}, []float64{1, 0, 0, 0, 1, 0, 1, 0})
	smallAlpha, smallKappa := bothCoefficients(t, small)
	smallGap := math.Abs(smallAlpha - smallKappa)

	const n = 20000
	large := matrix(coverage(n, 5000, 0), coverage(n, 5000, 1200))
	largeAlpha, largeKappa := bothCoefficients(t, large)
	largeGap := math.Abs(largeAlpha - largeKappa)

	t.Logf("8 items:     alpha %.4f kappa %.4f gap %.4f", smallAlpha, smallKappa, smallGap)
	t.Logf("%d items: alpha %.4f kappa %.4f gap %.4f", n, largeAlpha, largeKappa, largeGap)

	if smallGap <= largeGap {
		t.Errorf("expected the correction to matter more on a small matrix: %.4f vs %.4f",
			smallGap, largeGap)
	}
}
