import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var queue = OCRQueue()
    @State private var importing = false
    @State private var pickingFolder = false

    var body: some View {
        NavigationSplitView {
            SettingsPanel(queue: queue, pickFolder: { pickingFolder = true })
                .navigationSplitViewColumnWidth(min: 240, ideal: 280)
        } detail: {
            VStack(spacing: 0) {
                Toolbar(
                    addAction: { importing = true },
                    runAction: { queue.startAll() },
                    runDisabled: queue.jobs.isEmpty || queue.outputFolder == nil
                )
                Divider()
                JobList(queue: queue)
            }
        }
        .fileImporter(
            isPresented: $importing,
            allowedContentTypes: [.pdf],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result { queue.add(urls) }
        }
        .fileImporter(
            isPresented: $pickingFolder,
            allowedContentTypes: [.folder]
        ) { result in
            if case .success(let url) = result {
                queue.outputFolder = url
            }
        }
    }
}

private struct Toolbar: View {
    let addAction: () -> Void
    let runAction: () -> Void
    let runDisabled: Bool

    var body: some View {
        HStack {
            Button("Add PDFs…", systemImage: "plus", action: addAction)
            Button("Run All", systemImage: "play.fill", action: runAction)
                .disabled(runDisabled)
            Spacer()
        }
        .padding(8)
    }
}

private struct SettingsPanel: View {
    @Bindable var queue: OCRQueue
    let pickFolder: () -> Void

    var body: some View {
        Form {
            Section("Output") {
                Button {
                    pickFolder()
                } label: {
                    HStack {
                        Image(systemName: "folder")
                        Text(queue.outputFolder?.lastPathComponent ?? "Choose folder…")
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                Picker("Format", selection: $queue.format) {
                    ForEach(OutputFormat.allCases) { fmt in
                        Text(fmt.rawValue.capitalized).tag(fmt)
                    }
                }
            }
            Section("OCR") {
                LabeledContent("Render scale") {
                    HStack {
                        Slider(value: $queue.scale, in: 1.0...4.0, step: 0.5)
                        Text(String(format: "%.1f×", queue.scale))
                            .monospacedDigit()
                            .frame(width: 44, alignment: .trailing)
                    }
                }
                TextField("Languages (comma-separated)", text: $queue.languages)
            }
        }
        .formStyle(.grouped)
    }
}

private struct JobList: View {
    @Bindable var queue: OCRQueue

    var body: some View {
        if queue.jobs.isEmpty {
            ContentUnavailableView(
                "No PDFs",
                systemImage: "doc.text.magnifyingglass",
                description: Text("Add PDFs to start OCR.")
            )
        } else {
            List {
                ForEach(queue.jobs) { job in
                    JobRow(job: job, queue: queue)
                }
            }
        }
    }
}

private struct JobRow: View {
    @Bindable var job: OCRJob
    let queue: OCRQueue

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundStyle(tint)
            VStack(alignment: .leading, spacing: 2) {
                Text(job.url.lastPathComponent)
                    .lineLimit(1)
                Text(job.status.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if case .running(let p) = job.status {
                ProgressView(value: p).frame(width: 80)
                Button("Cancel") { queue.cancel(job) }
            } else if case .pending = job.status {
                Button("Run") { queue.start(job) }
                    .disabled(queue.outputFolder == nil)
            } else if case .done(let url) = job.status {
                Button("Reveal") {
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                }
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
