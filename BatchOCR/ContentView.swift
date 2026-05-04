import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var queue = OCRQueue()
    @State private var importing = false
    @State private var isTargeted = false
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(spacing: 16) {
            DropZone(isTargeted: $isTargeted) { importing = true }
                .frame(minHeight: 180)

            if !queue.jobs.isEmpty {
                Divider()
                JobList(queue: queue) { url in
                    openWindow(value: url)
                }
            }
        }
        .padding()
        .onAppear {
            queue.onJobReady = { url in openWindow(value: url) }
        }
        .fileImporter(
            isPresented: $importing,
            allowedContentTypes: [.pdf],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result { queue.add(urls) }
        }
        .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
            handleDrop(providers)
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        var urls: [URL] = []
        let group = DispatchGroup()
        for provider in providers {
            group.enter()
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url, url.pathExtension.lowercased() == "pdf" { urls.append(url) }
                group.leave()
            }
        }
        group.notify(queue: .main) {
            if !urls.isEmpty { queue.add(urls) }
        }
        return true
    }
}

private struct DropZone: View {
    @Binding var isTargeted: Bool
    let onClick: () -> Void

    var body: some View {
        Button(action: onClick) {
            VStack(spacing: 8) {
                Image(systemName: "doc.text.viewfinder")
                    .font(.system(size: 40))
                    .foregroundStyle(.secondary)
                Text("Drop PDFs here, or click to choose")
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(24)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(
                        isTargeted ? Color.accentColor : Color.secondary.opacity(0.4),
                        style: StrokeStyle(lineWidth: 2, dash: [6, 4])
                    )
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(isTargeted ? Color.accentColor.opacity(0.08) : .clear)
                    )
            )
        }
        .buttonStyle(.plain)
    }
}

private struct JobList: View {
    @Bindable var queue: OCRQueue
    let openResult: (URL) -> Void

    var body: some View {
        List {
            ForEach(queue.jobs) { job in
                JobRow(job: job, queue: queue, openResult: openResult)
            }
        }
        .frame(minHeight: 120)
    }
}

private struct JobRow: View {
    @Bindable var job: OCRJob
    let queue: OCRQueue
    let openResult: (URL) -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon).foregroundStyle(tint)
            VStack(alignment: .leading, spacing: 2) {
                Text(job.url.lastPathComponent).lineLimit(1)
                Text(job.status.label)
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if case .running(let p) = job.status {
                ProgressView(value: p).frame(width: 80)
                Button("Cancel") { queue.cancel(job) }
            } else if case .done(let url) = job.status {
                Button("Open") { openResult(url) }
            }
            Button {
                queue.remove(job)
            } label: {
                Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)
        }
        .padding(.vertical, 4)
    }

    private var icon: String {
        switch job.status {
        case .pending: return "doc"
        case .running: return "gearshape.2"
        case .done: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }
    private var tint: Color {
        switch job.status {
        case .pending: return .secondary
        case .running: return .blue
        case .done: return .green
        case .failed: return .orange
        }
    }
}
