import Foundation
import PDFKit

enum OutputFormat: String, CaseIterable, Identifiable, Codable {
    case text, markdown, json
    var id: String { rawValue }
    var fileExtension: String {
        switch self {
        case .text: return "txt"
        case .markdown: return "md"
        case .json: return "json"
        }
    }
}

struct OCRDocument: Codable {
    var sourceURL: URL
    var pages: [OCRPage]
}

enum PipelineError: Error {
    case cannotOpenPDF(URL)
    case renderFailed(pageIndex: Int)
}

struct PDFOCRPipeline {
    var scale: CGFloat = 3.0
    var languages: [String] = ["en-US"]

    func process(
        _ url: URL,
        progress: ((Int, Int) -> Void)? = nil
    ) async throws -> OCRDocument {
        guard let doc = PDFDocument(url: url) else {
            throw PipelineError.cannotOpenPDF(url)
        }
        let count = doc.pageCount
        var pages: [OCRPage] = []
        pages.reserveCapacity(count)
        for i in 0..<count {
            try Task.checkCancellation()
            guard let page = doc.page(at: i) else { continue }
            guard let image = PDFRenderer.render(page, scale: scale) else {
                throw PipelineError.renderFailed(pageIndex: i)
            }
            let lines = try await OCRService.recognize(image, languages: languages)
            pages.append(OCRPage(index: i, lines: lines))
            progress?(i + 1, count)
        }
        return OCRDocument(sourceURL: url, pages: pages)
    }
}

enum OCRWriter {
    static func encode(_ document: OCRDocument, as format: OutputFormat) throws -> Data {
        switch format {
        case .text:
            let body = document.pages
                .map { page in page.lines.map(\.text).joined(separator: "\n") }
                .joined(separator: "\n\n")
            return Data(body.utf8)
        case .markdown:
            var out = "# \(document.sourceURL.lastPathComponent)\n\n"
            for page in document.pages {
                out += "## Page \(page.index + 1)\n\n"
                out += page.lines.map(\.text).joined(separator: "\n")
                out += "\n\n"
            }
            return Data(out.utf8)
        case .json:
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            return try encoder.encode(document)
        }
    }
}
