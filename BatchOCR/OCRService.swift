import CoreGraphics
import Vision

struct TextLine: Sendable, Codable {
    var text: String
    var confidence: Float
    var box: CGRect
}

struct OCRPage: Sendable, Codable {
    var index: Int
    var lines: [TextLine]
}

enum OCRService {
    static func recognize(
        _ image: CGImage,
        languages: [String] = ["en-US"]
    ) async throws -> [TextLine] {
        var request = RecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.automaticallyDetectsLanguage = true
        request.recognitionLanguages = languages.map { Locale.Language(identifier: $0) }

        let observations = try await request.perform(on: image)
        return observations.compactMap { obs -> TextLine? in
            guard let top = obs.topCandidates(1).first else { return nil }
            return TextLine(
                text: top.string,
                confidence: top.confidence,
                box: obs.boundingBox.cgRect
            )
        }
    }
}
