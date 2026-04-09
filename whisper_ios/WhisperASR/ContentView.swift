import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var recorder = AudioRecorder()
    @StateObject private var downloader = ModelDownloader()
    @StateObject private var player = AudioPlayerManager()
    @State private var engine = WhisperEngine()
    @State private var status = ""
    @State private var result = ""
    @State private var selectedModel = 0
    @State private var showModelPicker = false
    @State private var isProcessing = false

    var body: some View {
        ZStack {
            Color.white.ignoresSafeArea()

            VStack(spacing: 0) {
                // Title
                Text("Whisper ASR")
                    .font(.system(size: 25, weight: .semibold))
                    .foregroundColor(Color(red: 0.01, green: 0.01, blue: 0.35))
                    .padding(.top, 24)

                // Status
                Text(recorder.isRecording
                     ? "Recording...\(recorder.recordingSeconds)s"
                     : status)
                    .font(.system(size: 14))
                    .foregroundColor(Color(white: 0.53))
                    .padding(.top, 8)
                    .padding(.horizontal, 20)

                // Transcription result
                ScrollView {
                    Text(result)
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundColor(Color(red: 0.01, green: 0.01, blue: 0.35))
                        .lineSpacing(6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(20)
                }
                .frame(maxHeight: .infinity)

                if downloader.isDownloading {
                    VStack(spacing: 4) {
                        ProgressView(value: downloader.progress)
                        Text("Downloading \(appModels[selectedModel].label) model (\(Int(downloader.progress * 100))%)...")
                            .font(.caption).foregroundColor(.gray)
                    }.padding(.horizontal, 20).padding(.bottom, 8)
                }

                // Audio playback bar (matches Android audioBar)
                if recorder.hasRecording && !recorder.isRecording {
                    HStack(spacing: 8) {
                        Button(action: { player.toggle(url: recorder.recordingURL) }) {
                            Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                                .foregroundColor(.gray)
                                .frame(width: 40, height: 40)
                        }

                        Slider(value: Binding(
                            get: { player.progress },
                            set: { player.seek(to: $0) }
                        ), in: 0...1)
                        .accentColor(Color(red: 0.01, green: 0.01, blue: 0.35))

                        Text(player.timeString)
                            .font(.system(size: 12))
                            .foregroundColor(Color(white: 0.53))
                            .frame(width: 80, alignment: .trailing)
                    }
                    .padding(.horizontal, 20)
                    .frame(height: 48)
                    .background(Color(white: 0.88))
                    .cornerRadius(10)
                    .padding(.horizontal, 16)
                }

                // Bottom button row (matches Android buttonRow)
                HStack {
                    // Model button
                    Button(appModels[selectedModel].label) { showModelPicker = true }
                        .font(.system(size: 11, weight: .bold))
                        .padding(.horizontal, 12).padding(.vertical, 10)
                        .background(Color(white: 0.46))
                        .foregroundColor(.white).clipShape(Capsule())
                        .disabled(isProcessing || downloader.isDownloading || recorder.isRecording)

                    Spacer()

                    // Mic / Stop button (80x80 like Android)
                    Button(action: toggleRecording) {
                        Image(systemName: recorder.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                            .resizable().frame(width: 80, height: 80)
                            .foregroundColor(recorder.isRecording ? .red : Color(red: 0.01, green: 0.01, blue: 0.35))
                    }.disabled(isProcessing || downloader.isDownloading || !engine.isLoaded)

                    Spacer()

                    // Copy button
                    Button(action: { UIPasteboard.general.string = result }) {
                        Image(systemName: "doc.on.doc")
                            .font(.system(size: 18))
                            .frame(width: 36, height: 36)
                            .background(Color(white: 0.46))
                            .foregroundColor(.white).clipShape(Circle())
                    }
                }
                .padding(.horizontal, 40)
                .padding(.bottom, 24)
            }
        }
        .confirmationDialog("Select Model", isPresented: $showModelPicker) {
            ForEach(0..<appModels.count, id: \.self) { i in
                Button(appModels[i].label) { selectModel(i) }
            }
        }
        .onAppear { loadCurrentModel() }
    }

    private func selectModel(_ index: Int) {
        selectedModel = index
        engine = WhisperEngine()
        result = ""
        loadCurrentModel()
    }

    private func loadCurrentModel() {
        let m = appModels[selectedModel]
        if let bundledPath = Bundle.main.path(forResource: m.fileName, ofType: "tflite") {
            loadEngine(modelPath: bundledPath, model: m); return
        }
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let downloadedPath = docsDir.appendingPathComponent(m.fileName + ".tflite").path
        if FileManager.default.fileExists(atPath: downloadedPath) {
            loadEngine(modelPath: downloadedPath, model: m); return
        }
        guard let url = m.downloadURL else { status = "Model not found"; return }
        status = "Downloading \(m.label) model..."
        downloader.download(url: url, fileName: m.fileName + ".tflite") { destURL in
            if let destURL = destURL { loadEngine(modelPath: destURL.path, model: m) }
            else { status = "Download failed" }
        }
    }

    private func loadEngine(modelPath: String, model: ModelConfig) {
        status = "Loading \(model.label) model..."
        guard let vocabPath = Bundle.main.path(forResource: model.vocabFile, ofType: "bin") else {
            status = "Vocab file not found"; return
        }
        DispatchQueue.global().async {
            do {
                let eng = WhisperEngine()
                try eng.loadModel(modelPath: modelPath, vocabPath: vocabPath, multilingual: model.multilingual)
                DispatchQueue.main.async { engine = eng; status = "\(model.label) ready" }
            } catch {
                DispatchQueue.main.async { status = "Load failed: \(error.localizedDescription)" }
            }
        }
    }

    private func toggleRecording() {
        if recorder.isRecording {
            recorder.stopRecording()
        } else {
            result = ""
            player.stop()
            recorder.startRecording { transcribe() }
        }
    }

    private func transcribe() {
        guard engine.isLoaded else { status = "Model not loaded"; return }
        player.loadFile(url: recorder.recordingURL)
        status = "Recording done...!"
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            isProcessing = true; status = "Processing..."
            let path = recorder.recordingURL.path
            DispatchQueue.global().async {
                let text = engine.transcribe(wavePath: path)
                DispatchQueue.main.async { result = text; status = "Processing done...!"; isProcessing = false }
            }
        }
    }
}

// Audio player with seek bar support
class AudioPlayerManager: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published var isPlaying = false
    @Published var progress: Double = 0
    @Published var duration: TimeInterval = 0
    private var audioPlayer: AVAudioPlayer?
    private var timer: Timer?

    var timeString: String {
        let cur = audioPlayer?.currentTime ?? 0
        return "\(fmt(cur)) / \(fmt(duration))"
    }

    private func fmt(_ t: TimeInterval) -> String {
        let s = Int(t)
        return "\(s / 60):\(String(format: "%02d", s % 60))"
    }

    func loadFile(url: URL) {
        if let p = try? AVAudioPlayer(contentsOf: url) {
            duration = p.duration
        }
    }

    func toggle(url: URL) {
        if isPlaying { stop(); return }
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.delegate = self
            audioPlayer?.play()
            isPlaying = true
            duration = audioPlayer?.duration ?? 0
            timer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
                guard let self = self, let p = self.audioPlayer else { return }
                self.progress = p.duration > 0 ? p.currentTime / p.duration : 0
            }
        } catch { print("Playback error: \(error)") }
    }

    func stop() {
        audioPlayer?.stop()
        timer?.invalidate()
        isPlaying = false
        progress = 0
    }

    func seek(to value: Double) {
        guard let p = audioPlayer else { return }
        p.currentTime = value * p.duration
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully: Bool) {
        DispatchQueue.main.async { self.stop() }
    }
}
