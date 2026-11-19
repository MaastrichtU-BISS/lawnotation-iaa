package iaa

// Whole-document labels: the labelset of a document-level task, and the
// second labelset of a hybrid one.
//
// A tag applied to a whole document has no extent, so there is nothing to
// match and no distribution within a document to describe. What is left is
// the question a presence matrix answers: did each annotator put this label
// on this document at all. That makes it the degenerate case of the span
// view rather than a separate kind of measurement — one unit per document
// instead of one per word.

// documentLabelResult scores one whole-document label. The span views are
// left empty on purpose; callers show them only when they carry something.
func documentLabelResult(documents []Document, label string, annotators []string) LabelResult {
	matrix := buildPresenceMatrix(documents, label, annotators)
	items := 0
	if len(matrix) > 0 {
		items = len(matrix[0])
	}
	// At this level a marked unit is a document, so the two counts coincide.
	marked := markedColumnCount(matrix)
	return LabelResult{
		SpanCountsPerAnnotator: spanCounts(documents, label, annotators),
		CoverageAgreement: CoverageAgreement{
			MatrixItems:        items,
			KrippendorffAlpha:  krippendorffAlpha(matrix),
			MarkedItems:        marked,
			AlphaPairs:         krippendorffAlphaAllPairs(matrix, annotators),
			CohenKappaPairs:    cohenKappaAllPairs(matrix, annotators),
			DocumentsWithLabel: marked,
		},
	}
}
