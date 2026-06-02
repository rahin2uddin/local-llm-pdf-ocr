# Bolt's Performance Journal ⚡

Critical learnings only - not a log.

---

## 2026-06-02 - N+1 Image Decode Pattern in Per-Box OCR

**Learning:** The `crop_for_ocr(image_b64, bbox)` function decoded the full page image from base64 for every single box. In dense-mode OCR (150+ boxes per page), this caused ~150 redundant base64 decodes + PIL `Image.open` calls per page.

**Impact:** Each decode costs ~50-200ms on a typical page image. A 150-box dense page wasted 7-30 seconds just on redundant I/O.

**Fix:** Created `crop_for_ocr_from_image(pil_image, bbox)` that takes a pre-decoded PIL Image. The caller decodes once and shares across all box crops. Updated `_ocr_per_box` and `_refine_uncertain` to use this pattern.

**Action:** When processing batches of items from the same source (images, files, database rows), look for decode/parse operations that happen per-item vs once-per-batch. The N+1 query problem applies to I/O too.

**Pattern:** "Decode once, crop many" - share expensive decode operations across batch processing.

## 2026-06-02 - Grounded PDF Rasterization Double JPEG Round Trip

**Learning:** The grounded PDF path encoded each PyMuPDF pixmap to JPEG, reopened
that JPEG with Pillow, then encoded a final thumbnail JPEG. Direct
`Image.frombytes` conversion preserves the final output while removing the
redundant lossy round trip. On the example PDFs this reduced rasterization by
~63-75ms/page (2.41x-3.59x).

**Action:** When moving image data between PyMuPDF and Pillow, pass raw pixmap
samples directly unless an encoded artifact is the actual boundary output.
