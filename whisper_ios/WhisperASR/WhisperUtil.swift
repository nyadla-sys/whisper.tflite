import Foundation

class WhisperUtil {
    static let WHISPER_SAMPLE_RATE = 16000
    static let WHISPER_N_FFT = 400
    static let WHISPER_N_MEL = 80
    static let WHISPER_HOP_LENGTH = 160
    static let WHISPER_CHUNK_SIZE = 30

    var vocab = WhisperVocab()
    var filters = WhisperFilter()
    var mel = WhisperMel()

    func getWordFromToken(token: Int) -> String? { vocab.tokenToWord[token] }

    func loadFiltersAndVocab(multilingual: Bool, vocabPath: String) throws -> Bool {
        vocab.setMultilingual(multilingual: multilingual)
        let data = try Data(contentsOf: URL(fileURLWithPath: vocabPath))
        var offset = 0

        let magic = readUInt32(from: data, at: &offset)
        guard magic == 0x5553454E else { print("Bad magic: \(magic)"); return false }

        filters.nMel = Int(readInt32(from: data, at: &offset))
        filters.nFft = Int(readInt32(from: data, at: &offset))
        print("[Whisper] nMel: \(filters.nMel), nFft: \(filters.nFft)")

        let filterCount = filters.nMel * filters.nFft
        filters.data = [Float](repeating: 0, count: filterCount)
        for i in 0..<filterCount {
            filters.data[i] = readFloat(from: data, at: &offset)
        }

        let nVocab = Int(readInt32(from: data, at: &offset))
        print("[Whisper] nVocab: \(nVocab)")
        for i in 0..<nVocab {
            let len = Int(readInt32(from: data, at: &offset))
            if let word = String(bytes: Array(data[offset..<(offset + len)]), encoding: .utf8) {
                vocab.tokenToWord[i] = word
            }
            offset += len
        }

        if vocab.isMultilingual() {
            vocab.tokenEot += 1; vocab.tokenSot += 1; vocab.tokenPrev += 1
            vocab.tokenSolm += 1; vocab.tokenNot += 1; vocab.tokenBeg += 1
        }

        for i in nVocab..<vocab.nVocab {
            let word: String
            let ti = Int32(i)
            switch ti {
            case vocab.tokenEot: word = "[_EOT_]"
            case vocab.tokenSot: word = "[_SOT_]"
            case vocab.tokenPrev: word = "[_PREV_]"
            case vocab.tokenNot: word = "[_NOT_]"
            case vocab.tokenBeg: word = "[_BEG_]"
            default:
                if ti > vocab.tokenBeg { word = "[_TT_\(i - Int(vocab.tokenBeg))]" }
                else { word = "[_extra_token_\(i)]" }
            }
            vocab.tokenToWord[i] = word
        }
        return true
    }

    private func readInt32(from data: Data, at offset: inout Int) -> Int32 {
        var val: Int32 = 0
        _ = withUnsafeMutableBytes(of: &val) { data.copyBytes(to: $0, from: offset..<(offset + 4)) }
        offset += 4
        return val
    }

    private func readUInt32(from data: Data, at offset: inout Int) -> UInt32 {
        var val: UInt32 = 0
        _ = withUnsafeMutableBytes(of: &val) { data.copyBytes(to: $0, from: offset..<(offset + 4)) }
        offset += 4
        return val
    }

    private func readFloat(from data: Data, at offset: inout Int) -> Float {
        var val: Float = 0
        _ = withUnsafeMutableBytes(of: &val) { data.copyBytes(to: $0, from: offset..<(offset + 4)) }
        offset += 4
        return val
    }

    func calculateMelSpectrogram(samples: [Float], nSamples: Int, nThreads: Int) -> [Float] {
        let fftSize = WhisperUtil.WHISPER_N_FFT
        let fftStep = WhisperUtil.WHISPER_HOP_LENGTH
        mel.nMel = WhisperUtil.WHISPER_N_MEL
        mel.nLen = nSamples / fftStep
        mel.data = [Float](repeating: 0, count: mel.nMel * mel.nLen)

        var hann = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize { hann[i] = Float(0.5 * (1.0 - cos(2.0 * Double.pi * Double(i) / Double(fftSize)))) }

        let nFft = 1 + fftSize / 2
        var fftIn = [Float](repeating: 0, count: fftSize)
        var fftOut = [Float](repeating: 0, count: fftSize * 2)

        for i in 0..<mel.nLen {
            let offset = i * fftStep
            for j in 0..<fftSize { fftIn[j] = offset + j < nSamples ? hann[j] * samples[offset + j] : 0 }
            fft(input: fftIn, output: &fftOut)
            for j in 0..<fftSize { fftOut[j] = fftOut[2*j] * fftOut[2*j] + fftOut[2*j+1] * fftOut[2*j+1] }
            for j in 1..<(fftSize/2) { fftOut[j] += fftOut[fftSize - j] }
            for j in 0..<mel.nMel {
                var sum: Double = 0
                for k in 0..<nFft { sum += Double(fftOut[k] * filters.data[j * nFft + k]) }
                if sum < 1e-10 { sum = 1e-10 }
                mel.data[j * mel.nLen + i] = Float(log10(sum))
            }
        }

        var mmax: Double = -1e20
        for v in mel.data { if Double(v) > mmax { mmax = Double(v) } }
        mmax -= 8.0
        for i in 0..<mel.data.count {
            if Double(mel.data[i]) < mmax { mel.data[i] = Float(mmax) }
            mel.data[i] = Float((Double(mel.data[i]) + 4.0) / 4.0)
        }
        return mel.data
    }

    private func fft(input: [Float], output: inout [Float]) {
        let n = input.count
        if n == 1 { output[0] = input[0]; output[1] = 0; return }
        if n % 2 == 1 { dft(input: input, output: &output); return }

        var even = [Float](repeating: 0, count: n/2)
        var odd = [Float](repeating: 0, count: n/2)
        for i in stride(from: 0, to: n, by: 2) { even[i/2] = input[i] }
        for i in stride(from: 1, to: n, by: 2) { odd[i/2] = input[i] }

        var evenOut = [Float](repeating: 0, count: n)
        var oddOut = [Float](repeating: 0, count: n)
        fft(input: even, output: &evenOut)
        fft(input: odd, output: &oddOut)

        for k in 0..<(n/2) {
            let theta = 2.0 * Float.pi * Float(k) / Float(n)
            let re = cos(theta), im = -sin(theta)
            let reOdd = oddOut[2*k], imOdd = oddOut[2*k+1]
            output[2*k] = evenOut[2*k] + re * reOdd - im * imOdd
            output[2*k+1] = evenOut[2*k+1] + re * imOdd + im * reOdd
            output[2*(k+n/2)] = evenOut[2*k] - re * reOdd + im * imOdd
            output[2*(k+n/2)+1] = evenOut[2*k+1] - re * imOdd - im * reOdd
        }
    }

    private func dft(input: [Float], output: inout [Float]) {
        let n = input.count
        for k in 0..<n {
            var re: Float = 0, im: Float = 0
            for i in 0..<n {
                let angle = 2.0 * Float.pi * Float(k) * Float(i) / Float(n)
                re += input[i] * cos(angle)
                im -= input[i] * sin(angle)
            }
            output[2*k] = re; output[2*k+1] = im
        }
    }
}

class WhisperVocab {
    var tokenEot: Int32 = 50256
    var tokenSot: Int32 = 50257
    var tokenPrev: Int32 = 50360
    var tokenSolm: Int32 = 50361
    var tokenNot: Int32 = 50362
    var tokenBeg: Int32 = 50363
    let tokenTranslate: Int32 = 50358
    let tokenTranscribe: Int32 = 50359
    var tokenToWord: [Int: String] = [:]
    var nVocab = 51865

    func setMultilingual(multilingual: Bool) { nVocab = multilingual ? 51865 : 51864 }
    func isMultilingual() -> Bool { nVocab == 51865 }
}

class WhisperFilter { var nMel = 0; var nFft = 0; var data: [Float] = [] }
class WhisperMel { var nLen = 0; var nMel = 0; var data: [Float] = [] }
