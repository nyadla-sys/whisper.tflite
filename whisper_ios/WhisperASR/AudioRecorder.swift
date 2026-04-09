import AVFoundation

class AudioRecorder: NSObject, ObservableObject, AVAudioRecorderDelegate {
    @Published var isRecording = false
    @Published var hasRecording = false
    @Published var recordingSeconds: Int = 0
    private var audioRecorder: AVAudioRecorder?
    private var completion: (() -> Void)?
    private var timer: Timer?
    static let maxDuration = 30

    private var cafURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("recording.caf")
    }

    var recordingURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("recording.wav")
    }

    func startRecording(completion: @escaping () -> Void) {
        self.completion = completion
        AVAudioSession.sharedInstance().requestRecordPermission { [weak self] allowed in
            DispatchQueue.main.async {
                if allowed { self?.beginRecording() }
            }
        }
    }

    private func beginRecording() {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)
            try? FileManager.default.removeItem(at: cafURL)
            try? FileManager.default.removeItem(at: recordingURL)

            audioRecorder = try AVAudioRecorder(url: cafURL, settings: settings)
            audioRecorder?.delegate = self
            audioRecorder?.prepareToRecord()
            audioRecorder?.record(forDuration: TimeInterval(Self.maxDuration))
            isRecording = true
            hasRecording = false
            recordingSeconds = 0

            timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
                guard let self = self else { return }
                self.recordingSeconds += 1
            }
        } catch {
            print("[Recorder] error: \(error)")
        }
    }

    func stopRecording() {
        audioRecorder?.stop()
        timer?.invalidate()
    }

    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        DispatchQueue.main.async {
            self.isRecording = false
            self.timer?.invalidate()
            if flag {
                // Read raw recorded data and write as proper WAV (same as reference project)
                if let recordedData = try? Data(contentsOf: recorder.url) {
                    let samples: [UInt8] = Array(recordedData)
                    WaveUtil.createWaveFile(
                        filePath: self.recordingURL.path,
                        samples: samples,
                        sampleRate: 16000,
                        numChannels: 1,
                        bytesPerSample: 2
                    )
                    print("[Recorder] WAV written: \(samples.count) bytes")
                    self.hasRecording = true
                    self.completion?()
                }
            }
        }
    }
}
