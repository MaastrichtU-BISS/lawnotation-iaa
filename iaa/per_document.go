package iaa

// Per-document coverage agreement, so a corpus-level alpha can be read
// against the spread of the documents it averages over.

// DocumentAlpha is one document's coverage agreement for one label.
type DocumentAlpha struct {
	Document string      `json:"document"`
	Alpha    NullFloat64 `json:"alpha"`
	// Kappa is defined only where there are exactly two annotators, which is
	// the case where a reader expects Cohen's kappa rather than a
	// multi-annotator coefficient.
	Kappa       NullFloat64 `json:"kappa"`
	MarkedItems int         `json:"marked_items"`
}

// perDocumentAlpha reports coverage agreement for each document in which the
// label was actually used.
//
// Documents where no annotator marked the label are left out, and the reason
// is not tidiness: over an all-zero matrix the observed and expected
// disagreement are both zero, so alpha comes back 1.0 — perfect agreement
// about nothing. A label used in forty documents out of nine hundred would
// otherwise show 860 dots at 1.0 and bury its own result.
func perDocumentAlpha(documents []Document, label string, annotators []string) []DocumentAlpha {
	var out []DocumentAlpha
	for _, doc := range documents {
		single := []Document{doc}
		matrix := buildCoverageMatrix(single, label, annotators)
		markedItems := markedColumnCount(matrix)
		if markedItems == 0 {
			continue
		}
		kappa := noFloat()
		if len(matrix) == 2 {
			kappa = cohenKappaPair(matrix[0], matrix[1])
		}
		out = append(out, DocumentAlpha{
			Document:    doc.Name,
			Alpha:       krippendorffAlpha(matrix),
			Kappa:       kappa,
			MarkedItems: markedItems,
		})
	}
	return out
}

// perDocumentPresence is the document-level equivalent: with no span extent
// there is nothing to measure within a single document, so there is no
// per-document view to report.
func perDocumentPresence() []DocumentAlpha { return nil }
