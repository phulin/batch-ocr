import Foundation

/// Minimal `.npy` reader for the artifact-comparison harness.
/// Supports little-endian float32 and float64 arrays, any rank.
public enum NPY {
    public struct Array {
        public var shape: [Int]
        public var floats: [Float]   // Always materialized as Float32 for downstream comparisons.
    }

    public enum Error: Swift.Error {
        case badMagic
        case unsupportedDtype(String)
        case truncated
    }

    public static func read(_ url: URL) throws -> Array {
        let data = try Data(contentsOf: url)
        guard data.count >= 10 else { throw Error.truncated }

        // Magic: "\x93NUMPY"
        let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]
        guard Swift.Array(data[0..<6]) == magic else { throw Error.badMagic }

        let major = data[6]
        let _ = data[7]
        let headerLenSize = (major >= 2) ? 4 : 2
        var headerLen = 0
        for i in 0..<headerLenSize {
            headerLen |= Int(data[8 + i]) << (8 * i)
        }
        let headerStart = 8 + headerLenSize
        let headerEnd = headerStart + headerLen
        let header = String(data: data[headerStart..<headerEnd], encoding: .ascii) ?? ""

        // Parse a few key=val pairs from the header dict — naive but enough for our needs.
        let descr = parseField(header: header, key: "descr") ?? "<f4"
        let shapeStr = parseField(header: header, key: "shape") ?? "()"
        let shape = parseShape(shapeStr)

        let payload = data[headerEnd...]
        let floats: [Float]
        switch descr {
        case "<f4":
            floats = payload.withUnsafeBytes { buf in
                Swift.Array(buf.bindMemory(to: Float.self))
            }
        case "<f8":
            floats = payload.withUnsafeBytes { buf in
                buf.bindMemory(to: Double.self).map { Float($0) }
            }
        case "<i4":
            floats = payload.withUnsafeBytes { buf in
                buf.bindMemory(to: Int32.self).map { Float($0) }
            }
        case "<i8":
            floats = payload.withUnsafeBytes { buf in
                buf.bindMemory(to: Int64.self).map { Float($0) }
            }
        default:
            throw Error.unsupportedDtype(descr)
        }
        return Array(shape: shape, floats: floats)
    }

    private static func parseField(header: String, key: String) -> String? {
        // looks for: 'key': '...'  OR 'key': (...)  OR 'key': True/False
        guard let range = header.range(of: "'\(key)':") else { return nil }
        let after = header[range.upperBound...].drop(while: { $0 == " " })
        if let first = after.first {
            switch first {
            case "'":
                let body = after.dropFirst()
                if let end = body.firstIndex(of: "'") { return String(body[..<end]) }
            case "(":
                let body = after.dropFirst()
                if let end = body.firstIndex(of: ")") { return "(" + body[..<end] + ")" }
            case "T", "F":
                if after.hasPrefix("True") { return "True" }
                if after.hasPrefix("False") { return "False" }
            default:
                let s = after.prefix(while: { $0 != "," && $0 != "}" })
                return s.trimmingCharacters(in: .whitespaces)
            }
        }
        return nil
    }

    private static func parseShape(_ s: String) -> [Int] {
        let inner = s.trimmingCharacters(in: CharacterSet(charactersIn: "() "))
        if inner.isEmpty { return [] }
        return inner.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .compactMap { Int($0) }
    }
}
