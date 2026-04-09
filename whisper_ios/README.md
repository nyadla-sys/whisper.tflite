# Whisper iOS — Offline Speech Recognition

Offline speech recognition for iOS using OpenAI Whisper quantized TFLite models. Models are downloaded on-demand from HuggingFace to keep the app size small.

## Prerequisites

- Xcode 15 or newer
- Apple Developer account (free for device testing, $99/year for App Store/TestFlight)
- iOS device with iOS 16+ or simulator

## Build Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/nyadla-sys/whisper.tflite.git
   cd whisper.tflite/whisper_ios
   ```

2. Open the project in Xcode:
   ```bash
   open WhisperASR.xcodeproj
   ```

3. Xcode will automatically resolve the TFLite Swift Package dependency from [kewlbear/TensorFlowLiteSwift](https://github.com/kewlbear/TensorFlowLiteSwift).

4. Select your signing team in **Signing & Capabilities**.

5. Build and run:
   - **⌘R** to run on a connected device or simulator
   - **⌘B** to build without running

## Models

The app includes a model selection menu with three options:

| Model | Size | Language | Bundled in App |
|---|---|---|---|
| `tiny.en` | 40 MB | English only | ✅ Yes |
| `small.en` | 247 MB | English only | ❌ Downloaded on first use |
| `small` | 249 MB | Multilingual | ❌ Downloaded on first use |

- **tiny.en** is bundled in the app and works offline immediately.
- **small.en** and **small** are downloaded from [HuggingFace](https://huggingface.co/nyadla-sys/whisper-tiny.en.tflite) when the user selects them. Download progress is shown in the UI.
- Downloaded models are cached in the app's Documents directory for future use.

### Vocab files

Vocab files are bundled in the app:

| Vocab file | Used by |
|---|---|
| `filters_vocab_en.bin` | tiny.en, small.en |
| `filters_vocab_multilingual.bin` | small (multilingual) |

## Permissions

- **Microphone** — required for recording audio (prompted on first use)

## Architecture

The app is built with SwiftUI and uses the following components:

| File | Description |
|---|---|
| `WhisperASRApp.swift` | App entry point |
| `ContentView.swift` | Main UI — model picker, mic button, transcription display, audio playback bar |
| `WhisperEngine.swift` | TFLite interpreter, mel spectrogram input, Int32 token output decoding |
| `WhisperUtil.swift` | Mel spectrogram computation, FFT, vocab/filter binary loading |
| `WaveUtil.swift` | WAV file creation (RIFF header) and parsing (16-bit and 32-bit PCM) |
| `AudioRecorder.swift` | Mic recording via AVAudioRecorder with CAF-to-WAV conversion |
| `ModelDownloader.swift` | On-demand model download from HuggingFace with progress |

## Features

- **Recording timer** — status shows `Recording...0s`, `Recording...1s`, etc. up to 30s
- **Audio playback bar** — play/pause, seek slider, and duration display after recording
- **Status messages** — `Recording done...!` → `Processing...` → `Processing done...!`
- **Model selection** — tap the model button to switch between tiny.en, small.en, and small
- **Copy to clipboard** — tap the copy button to copy the transcription result

## TestFlight / App Store Deployment

1. **Archive**: `Product → Archive` in Xcode (or `xcodebuild archive` from CLI)
2. **Fix TFLite plists** (CLI builds only): patch `MinimumOSVersion` and `CFBundleShortVersionString` in the TFLite framework Info.plist files inside the archive
3. **Upload**: `Window → Organizer` → select archive → `Distribute App` → TestFlight & App Store
4. **Install**: Open the TestFlight app on your iPhone and install the build
