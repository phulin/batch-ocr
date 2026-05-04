import SwiftUI

@main
struct BatchOCRApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 520, minHeight: 360)
        }

        WindowGroup(for: URL.self) { $url in
            if let url {
                PDFViewerWindow(url: url)
            }
        }
    }
}
