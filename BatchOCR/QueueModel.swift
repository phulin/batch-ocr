import Foundation
import Observation

enum JobStatus: Equatable {
    case pending
    case running(Double)
    case done(URL)
    case failed(String)

    var label: String {
        switch self {
        case .pending: return "Pending"
        case .running(let p): return "OCR \(Int(p * 100))%"
        case .done: return "Done"
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
    var format: OutputFormat = .markdown
    var outputFolder: URL?
    var languages: String = "en-US"

    func add(_ urls: [URL]) {
        let existing = Set(jobs.map(\.url))
        for url in urls where !existing.contains(url) {
            jobs.append(OCRJob(url: url))
        }
    }

    func remove(_ job: OCRJob) {
        job.task?.cancel()
        jobs.removeAll { $0.id == job.id }
    }

    func start(_ job: OCRJob) {
        guard case .pending = job.status else { return }
        guard let outputFolder else {
            job.status = .failed("No output folder selected")
            return
        }
        let pipeline = PDFOCRPipeline(
            scale: CGFloat(scale),
            languages: languages
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
        )
        let format = format
        job.status = .running(0)

        let needsScope = outputFolder.startAccessingSecurityScopedResource()
        let inputScope = job.url.startAccessingSecurityScopedResource()

        job.task = Task { [weak job] in
            defer {
                if needsScope { outputFolder.stopAccessingSecurityScopedResource() }
                if inputScope { job?.url.stopAccessingSecurityScopedResource() }
            }
            guard let job else { return }
            do {
                let doc = try await pipeline.process(job.url) { done, total in
                    Task { @MainActor in
                        job.status = .running(Double(done) / Double(total))
                    }
                }
                let data = try OCRWriter.encode(doc, as: format)
                let outURL = outputFolder
                    .appendingPathComponent(job.url.deletingPathExtension().lastPathComponent)
                    .appendingPathExtension(format.fileExtension)
                try data.write(to: outURL, options: .atomic)
                await MainActor.run { job.status = .done(outURL) }
            } catch is CancellationError {
                await MainActor.run { job.status = .failed("Cancelled") }
            } catch {
                await MainActor.run { job.status = .failed(error.localizedDescription) }
            }
        }
    }

    func startAll() {
        for job in jobs { start(job) }
    }

    func cancel(_ job: OCRJob) {
        job.task?.cancel()
    }
}
