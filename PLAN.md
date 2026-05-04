# batch-ocr — macOS PDF OCR (Apple Vision)

Native macOS SwiftUI app to OCR PDFs in batch using Apple's Vision framework. (PaddleOCR-VL deferred — see "Future" below.)

## Stack

- **OCR**: Vision `RecognizeTextRequest` (macOS 15+/Xcode 26 async API). Replaces `VNRecognizeTextRequest`. Docs: https://developer.apple.com/documentation/vision/recognizetextrequest
- **PDF rendering**: PDFKit, `PDFPage.draw(with:to:)` into a `CGContext` at 2–3× scale for high-DPI input.
- **UI**: SwiftUI app, `.fileImporter` for input.
- **Project gen**: `xcodegen` (installed, 2.45.4) with `project.yml`.

## Vision API shape

```swift
var request = RecognizeTextRequest()
request.recognitionLevel = .accurate
request.automaticallyDetectsLanguage = true
request.usesLanguageCorrection = true
let observations = try await request.perform(on: cgImage)
for obs in observations {
    guard let top = obs.topCandidates(1).first else { continue }
    let text = top.string
    let box = obs.boundingBox.cgRect   // normalized, bottom-left origin
}
```

## Entitlements (sandboxed)

- `com.apple.security.app-sandbox` = YES
- `com.apple.security.files.user-selected.read-write` = YES

## Phases

### Phase 1 — Scaffold
- [ ] `project.yml` for xcodegen → SwiftUI macOS app target, deploys 15.0+
- [ ] `.entitlements`, `Info.plist`
- [ ] `xcodegen generate` produces `BatchOCR.xcodeproj`
- [ ] Smoke build via `xcodebuild`

### Phase 2 — OCR core
- [ ] `PDFRenderer`: `PDFDocument` → per-page `CGImage` at configurable scale (default 3.0)
- [ ] `OCRService`: async `recognize(_ image: CGImage) -> [TextLine]` wrapping `RecognizeTextRequest`
- [ ] `PDFOCRPipeline`: file URL → `[Page { lines: [TextLine] }]`
- [ ] Output writers: plain text, Markdown (one section per page), JSON with bounding boxes

### Phase 3 — UI
- [ ] Drag-and-drop + `.fileImporter` for PDFs
- [ ] Queue list with per-file status: pending / running / done / failed
- [ ] Settings: render scale, output format, output folder, recognition languages
- [ ] Progress + cancel per file

### Phase 4 — Polish
- [ ] Concurrency: process pages in parallel within a file (`TaskGroup`); cap files-in-flight
- [ ] Bookmark persistence for output folder across launches (`com.apple.security.files.bookmarks.app-scope` if needed)
- [ ] Error surfacing in the queue row

## Future — swap in PaddleOCR-VL

Vision handles plain text well but can't do tables/formulas/charts as structure. When Vision proves insufficient, swap `OCRService` for the [`mlx-community/paddleocr-vl.swift`](https://github.com/mlx-community/paddleocr-vl.swift) port; the pipeline boundary (`CGImage → [TextLine | Region]`) is designed to make that swap mechanical. Reference repos kept in `refs/` (gitignored).
