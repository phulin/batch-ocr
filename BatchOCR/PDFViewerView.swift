import AppKit
import PDFKit
import SwiftUI

struct PDFViewerView: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> PDFView {
        let view = PDFView()
        view.autoScales = true
        view.displayMode = .singlePageContinuous
        view.displayDirection = .vertical
        view.document = PDFDocument(url: url)
        return view
    }

    func updateNSView(_ view: PDFView, context: Context) {
        if view.document?.documentURL != url {
            view.document = PDFDocument(url: url)
        }
    }
}

struct PDFViewerWindow: View {
    let url: URL

    var body: some View {
        PDFViewerView(url: url)
            .frame(minWidth: 600, minHeight: 700)
            .navigationTitle(url.lastPathComponent)
    }
}
