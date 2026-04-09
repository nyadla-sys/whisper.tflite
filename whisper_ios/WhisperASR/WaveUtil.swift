import Foundation

class WaveUtil {
    static func createWaveFile(filePath: String, samples: [UInt8], sampleRate: Int, numChannels: Int, bytesPerSample: Int) {
        let dataSize = samples.count
        let audioFormat = bytesPerSample == 2 ? 1 : (bytesPerSample == 4 ? 3 : 0)
        guard let file = FileHandle(forWritingAtPath: filePath) ?? {
            FileManager.default.createFile(atPath: filePath, contents: nil)
            return FileHandle(forWritingAtPath: filePath)
        }() else { return }

        file.write("RIFF".data(using: .utf8)!)
        file.write(intBytes(36 + dataSize))
        file.write("WAVE".data(using: .utf8)!)
        file.write("fmt ".data(using: .utf8)!)
        file.write(intBytes(16))
        file.write(shortBytes(audioFormat))
        file.write(shortBytes(numChannels))
        file.write(intBytes(sampleRate))
        file.write(intBytes(sampleRate * numChannels * bytesPerSample))
        file.write(shortBytes(numChannels * bytesPerSample))
        file.write(shortBytes(bytesPerSample * 8))
        file.write("data".data(using: .utf8)!)
        file.write(intBytes(dataSize))
        file.write(Data(samples))
        file.closeFile()
    }

    private static func intBytes(_ value: Int) -> Data {
        var v = UInt32(value)
        return Data(bytes: &v, count: 4)
    }

    private static func shortBytes(_ value: Int) -> Data {
        var v = UInt16(value)
        return Data(bytes: &v, count: 2)
    }

    static func getSamples(from filePath: String) -> [Float] {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)),
              data.count > 44 else { return [] }

        let header = String(data: data[0..<4], encoding: .ascii) ?? ""
        guard header == "RIFF" else { print("[WaveUtil] Not RIFF"); return [] }

        // Read bitsPerSample safely (offset 34, 2 bytes)
        var bitsPerSample: UInt16 = 0
        _ = withUnsafeMutableBytes(of: &bitsPerSample) { data.copyBytes(to: $0, from: 34..<36) }
        print("[WaveUtil] bitsPerSample: \(bitsPerSample)")

        // Find "data" chunk
        var offset = 12
        while offset + 8 < data.count {
            let chunkID = String(data: data[offset..<(offset+4)], encoding: .ascii) ?? ""
            var chunkSize: UInt32 = 0
            _ = withUnsafeMutableBytes(of: &chunkSize) { data.copyBytes(to: $0, from: (offset+4)..<(offset+8)) }
            if chunkID == "data" {
                let dataStart = offset + 8
                let dataEnd = min(dataStart + Int(chunkSize), data.count)
                print("[WaveUtil] data chunk at \(dataStart), size: \(dataEnd - dataStart)")
                return convertSamples(data: data, from: dataStart, to: dataEnd, bitsPerSample: Int(bitsPerSample))
            }
            offset += 8 + Int(chunkSize)
        }

        // Fallback: skip 44-byte header
        print("[WaveUtil] No data chunk found, using offset 44")
        return convertSamples(data: data, from: 44, to: data.count, bitsPerSample: Int(bitsPerSample))
    }

    private static func convertSamples(data: Data, from start: Int, to end: Int, bitsPerSample: Int) -> [Float] {
        var samples: [Float] = []
        var offset = start

        if bitsPerSample == 16 {
            samples.reserveCapacity((end - start) / 2)
            while offset + 1 < end {
                var val: Int16 = 0
                _ = withUnsafeMutableBytes(of: &val) { data.copyBytes(to: $0, from: offset..<(offset+2)) }
                samples.append(Float(val) / 32768.0)
                offset += 2
            }
        } else if bitsPerSample == 32 {
            samples.reserveCapacity((end - start) / 4)
            while offset + 3 < end {
                var val: Float = 0
                _ = withUnsafeMutableBytes(of: &val) { data.copyBytes(to: $0, from: offset..<(offset+4)) }
                samples.append(val)
                offset += 4
            }
        }
        print("[WaveUtil] Parsed \(samples.count) samples")
        return samples
    }
}
