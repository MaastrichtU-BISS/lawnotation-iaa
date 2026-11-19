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

// documentTagView returns the documents with each assignment's hybrid
// document tags in place of its spans, as annotations with no extent, which
// is how a document-level task carries its tags. Scoring the tags this way
// keeps a document label apart from a span label of the same name.
func documentTagView(documents []Document) []Document {
	out := make([]Document, len(documents))
	for i, doc := range documents {
		view := doc
		view.Assignments = make([]Assignment, len(doc.Assignments))
		for j, asgn := range doc.Assignments {
			tags := asgn
			tags.Annotations = make([]Annotation, len(asgn.DocumentAnnotations))
			for k, tag := range asgn.DocumentAnnotations {
				tags.Annotations[k] = Annotation{Label: tag.Label}
			}
			view.Assignments[j] = tags
		}
		out[i] = view
	}
	return out
}
