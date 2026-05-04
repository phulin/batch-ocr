import Foundation
import Observation
import PDFKit

enum JobStatus: Equatable {
    case pending
    case running(Double)
    case done(URL)
    case failed(String)

    var label: String {
        switch self {
        case .pending: return "Pending"
        case .running(let p): return "OCR \(Int(p * 100))%"
        case .done: return "Ready"
        case .failed(let msg): return "Failed: \(msg)"
        }
    }
}

@Observable
final class OCRJob: Identifiable {
    let id = UUID()
    let url: URL
    var status: JobStatus = .pending
    var task: Task<Void, Never>?

    init(url: URL) { self.url = url }
}

@MainActor
@Observable
final class OCRQueue {
    var jobs: [OCRJob] = []
    var scale: Double = 3.0
    var languages: String = "en-US"

    /// Called on the main actor when a job finishes successfully with the searchable-PDF URL.
    var onJobReady: ((URL) -> Void)?

    private let outputDirectory: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let dir = base.appendingPathComponent("BatchOCR/Searchable", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    func add(_ urls: [URL], autoStart: Bool = true) {
        let existing = Set(jobs.map(\.url))
        for url in urls where !existing.contains(url) {
            let job = OCRJob(url: url)
            jobs.append(job)
            if autoStart { start(job) }
        }
    }

    func remove(_ job: OCRJob) {
        job.task?.cancel()
        jobs.removeAll { $0.id == job.id }
    }

    func start(_ job: OCRJob) {
        guard case .pending = job.status else { return }
        let pipeline = PDFOCRPipeline(
            scale: CGFloat(scale),
            languages: languages
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
        )
        job.status = .running(0)

        let inputScope = job.url.startAccessingSecurityScopedResource()
        let outDir = outputDirectory
        let onReady = onJobReady

        job.task = Task { [weak job] in
            defer {
                if inputScope { job?.url.stopAccessingSecurityScopedResource() }
            }
            guard let job else { return }
            do {
                let doc = try await pipeline.process(job.url) { done, total in
                    Task { @MainActor in
                        job.status = .running(Double(done) / Double(total))
                    }
                }
                guard let pdf = PDFDocument(url: job.url) else {
                    throw PipelineError.cannotOpenPDF(job.url)
                }
                let outURL = outDir
                    .appendingPathComponent(job.url.deletingPathExtension().lastPathComponent + "-ocr")
                    .appendingPathExtension("pdf")
                try SearchablePDFWriter.write(source: pdf, ocrPages: doc.pages, to: outURL)
                await MainActor.run {
                    job.status = .done(outURL)
                    onReady?(outURL)
                }
            } catch is CancellationError {
                await MainActor.run { job.status = .failed("Cancelled") }
            } catch {
                await MainActor.run { job.status = .failed(error.localizedDescription) }
            }
        }
    }

    func cancel(_ job: OCRJob) {
        job.task?.cancel()
    }
}
