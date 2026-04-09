import Foundation
import TensorFlowLite

struct ModelConfig {
    let label: String
    let fileName: String
    let vocabFile: String
    let multilingual: Bool
    let downloadURL: String?
}

let appModels = [
    ModelConfig(label: "tiny.en", fileName: "whisper-tiny-en", vocabFile: "filters_vocab_en", multilingual: false, downloadURL: nil),
    ModelConfig(label: "small.en", fileName: "whisper-small.en", vocabFile: "filters_vocab_en", multilingual: false, downloadURL: "https://huggingface.co/nyadla-sys/whisper-tiny.en.tflite/resolve/main/whisper-small.en.tflite"),
    ModelConfig(label: "small", fileName: "whisper-small", vocabFile: "filters_vocab_multilingual", multilingual: true, downloadURL: "https://huggingface.co/nyadla-sys/whisper-tiny.en.tflite/resolve/main/whisper-small.tflite"),
]

class WhisperEngine {
    let whisperUtil = WhisperUtil()
    var interpreter: Interpreter?
    var isLoaded: Bool { interpreter != nil }

    func loadModel(modelPath: String, vocabPath: String, multilingual: Bool) throws {
        var options = Interpreter.Options()
        options.threadCount = ProcessInfo.processInfo.activeProcessorCount
        interpreter = try Interpreter(modelPath: modelPath, options: options)
        try interpreter?.allocateTensors()
        let _ = try whisperUtil.loadFiltersAndVocab(multilingual: multilingual, vocabPath: vocabPath)
    }

    func transcribe(wavePath: String) -> String {
        guard let interpreter = interpreter else { return "Model not loaded" }

        print("[Whisper] Transcribing: \(wavePath)")
        let samples = WaveUtil.getSamples(from: wavePath)
        print("[Whisper] Samples: \(samples.count)")
        guard samples.count > 0 else { return "Error: no audio samples" }

        let fixedSize = WhisperUtil.WHISPER_SAMPLE_RATE * WhisperUtil.WHISPER_CHUNK_SIZE
        var padded = [Float](repeating: 0, count: fixedSize)
        let copyLen = min(samples.count, fixedSize)
        padded[0..<copyLen] = samples[0..<copyLen]

        let cores = ProcessInfo.processInfo.processorCount
        print("[Whisper] Computing mel spectrogram...")
        let melData = whisperUtil.calculateMelSpectrogram(samples: padded, nSamples: padded.count, nThreads: cores)
        print("[Whisper] Mel size: \(melData.count)")
        let inputBuffer = melData.withUnsafeBufferPointer { Data(buffer: $0) }

        do {
            try interpreter.copy(inputBuffer, toInputAt: 0)
            print("[Whisper] Running inference...")
            try interpreter.invoke()
            print("[Whisper] Inference done")

            let outputTensor = try interpreter.output(at: 0)
            let outputSize = outputTensor.shape.dimensions.reduce(1, { $0 * $1 })
            print("[Whisper] Output size: \(outputSize), shape: \(outputTensor.shape.dimensions)")
            let outputData = UnsafeMutableBufferPointer<Int32>.allocate(capacity: outputSize)
            let _ = outputTensor.data.copyBytes(to: outputData)

            // Print first 20 tokens for debugging
            let tokenPreview = (0..<min(20, outputSize)).map { "\(outputData[$0])" }.joined(separator: ", ")
            print("[Whisper] First tokens: \(tokenPreview)")

            var result = ""
            for i in 0..<outputSize {
                let token = outputData[i]
                if token == whisperUtil.vocab.tokenEot { break }
                if token < whisperUtil.vocab.tokenEot,
                   let word = whisperUtil.getWordFromToken(token: Int(token)) {
                    result.append(word)
                }
            }
            outputData.deallocate()
            print("[Whisper] Raw result: \(result)")

            let cleaned = result
                .replacingOccurrences(of: #"<\|[^|]*\|>"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: #"\[_[A-Z]+_\]"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: #"\[_TT_\d+\]"#, with: "", options: .regularExpression)
                .trimmingCharacters(in: .whitespaces)
            print("[Whisper] Cleaned: \(cleaned)")
            return cleaned
        } catch {
            return "Inference error: \(error.localizedDescription)"
        }
    }
}
