import CoreGraphics
import CoreText
import Foundation
import PDFKit

enum SearchablePDFWriter {
    enum WriterError: Error {
        case cannotCreateContext(URL)
        case noPages
    }

    static func write(
        source: PDFDocument,
        ocrPages: [OCRPage],
        to outputURL: URL
    ) throws {
        let pageCount = source.pageCount
        guard pageCount > 0 else { throw WriterError.noPages }

        guard let ctx = CGContext(outputURL as CFURL, mediaBox: nil, nil) else {
            throw WriterError.cannotCreateContext(outputURL)
        }

        let ocrByIndex = Dictionary(uniqueKeysWithValues: ocrPages.map { ($0.index, $0) })

        for i in 0..<pageCount {
            guard let page = source.page(at: i) else { continue }
            let box = page.bounds(for: .mediaBox)
            ctx.beginPDFPage([kCGPDFContextMediaBox as String: NSValue(rect: box)] as CFDictionary)

            ctx.saveGState()
            page.draw(with: .mediaBox, to: ctx)
            ctx.restoreGState()

            if let ocr = ocrByIndex[i] {
                drawInvisibleText(ocr.lines, in: box, ctx: ctx)
            }

            ctx.endPDFPage()
        }
        ctx.closePDF()
    }

    private static func drawInvisibleText(
        _ lines: [TextLine],
        in pageBox: CGRect,
        ctx: CGContext
    ) {
        ctx.saveGState()
        ctx.setTextDrawingMode(.invisible)

        for line in lines where !line.text.isEmpty {
            // MinerU bboxes are normalized 0…1 with origin top-left. PDF user space origin is
            // bottom-left, so flip y. Mapping: pdf_y_bottom = pageBox.height * (1 - top_left_y2).
            let bx = line.box.minX
            let by = line.box.minY        // top-left y of bbox
            let bw = line.box.width
            let bh = line.box.height
            let rect = CGRect(
                x: pageBox.minX + bx * pageBox.width,
                y: pageBox.minY + (1.0 - (by + bh)) * pageBox.height,
                width: bw * pageBox.width,
                height: bh * pageBox.height
            )
            guard rect.height > 0.5, rect.width > 0.5 else { continue }

            // Pick a font size so the rendered glyph run roughly matches the box width;
            // selection accuracy depends on this — too small and clicks miss, too big and
            // selection overshoots.
            let probeSize: CGFloat = max(1, rect.height * 0.8)
            let font = CTFontCreateWithName("Helvetica" as CFString, probeSize, nil)
            let attrs: [NSAttributedString.Key: Any] = [.font: font]
            let attributed = NSAttributedString(string: line.text, attributes: attrs)
            let ctLine = CTLineCreateWithAttributedString(attributed)
            let typoBounds = CTLineGetBoundsWithOptions(ctLine, .useOpticalBounds)
            let scale = typoBounds.width > 0 ? rect.width / typoBounds.width : 1.0

            ctx.saveGState()
            ctx.textMatrix = CGAffineTransform(scaleX: scale, y: 1.0)
            ctx.textPosition = CGPoint(x: rect.minX, y: rect.minY)
            CTLineDraw(ctLine, ctx)
            ctx.restoreGState()
        }

        ctx.restoreGState()
    }
}
