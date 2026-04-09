# Whisper Android — Offline Speech Recognition

Offline speech recognition for Android using OpenAI Whisper quantized TFLite models. Models are downloaded on-demand from HuggingFace to keep the APK small.

## Prerequisites

- Android Studio Hedgehog or newer
- Android NDK (installed via SDK Manager)
- CMake 3.22+ (installed via SDK Manager)
- Android device or emulator with API 26+

## Build Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/nyadla-sys/whisper.tflite.git
   cd whisper.tflite/whisper_android
   ```

2. Open the `whisper_android` folder in Android Studio (File → Open).

3. Let Gradle sync complete. The project uses:
   - AGP 8.2.2
   - `compileSdk 34`, `minSdk 26`
   - Supported ABIs: `armeabi-v7a`, `arm64-v8a`

4. Pre-built TFLite `.so` libraries are already included under:
   ```
   app/src/main/cpp/tf-lite-api/generated-libs/
   ├── arm64-v8a/libtensorflowlite.so
   ├── armeabi-v7a/libtensorflowlite.so
   └── x86_64/libtensorflowlite.so
   ```
   No separate download is needed.

5. Build a release AAB for Play Store:
   ```bash
   ./gradlew bundleRelease
   ```
   Output: `app/build/outputs/bundle/release/app-release.aab`

   Or build a release APK:
   ```bash
   ./gradlew assembleRelease
   ```
   Output: `app/build/outputs/apk/release/app-release.apk`

## Running Tests

JVM property-based tests (jqwik 1.8.4):
```bash
./gradlew test
```

## Models

The app includes a model selection menu with three options:

| Model | Size | Language | Bundled in APK |
|---|---|---|---|
| `tiny.en` | 40 MB | English only | ✅ Yes |
| `small.en` | 247 MB | English only | ❌ Downloaded on first use |
| `small` | 249 MB | Multilingual | ❌ Downloaded on first use |

- **tiny.en** is bundled in the APK and works offline immediately.
- **small.en** and **small** are downloaded from [HuggingFace](https://huggingface.co/nyadla-sys/whisper-tiny.en.tflite) when the user selects them. Download progress is shown in the status bar.
- Downloaded models are cached in the app's data directory for future use.
- Use the English-only models for better performance when you only need English transcription.
- Use the **small** (multilingual) model when transcribing other languages.

### Vocab files

Vocab files are bundled in `app/src/main/assets/`:

| Vocab file | Used by |
|---|---|
| `filters_vocab_en.bin` | tiny.en, small.en |
| `filters_vocab_multilingual.bin` | small (multilingual) |

## Permissions

- **Microphone** — required for recording audio
- **Internet** — required for downloading models on first use

## Whisper API Integration

### Initialization

```java
Whisper mWhisper = new Whisper(this);

// English only
String modelPath = getFilePath("whisper-tiny-en.tflite");
String vocabPath = getFilePath("filters_vocab_en.bin");
mWhisper.loadModel(modelPath, vocabPath, false); // false = English only

// Multilingual
// String modelPath = getFilePath("whisper-small.tflite");
// String vocabPath = getFilePath("filters_vocab_multilingual.bin");
// mWhisper.loadModel(modelPath, vocabPath, true); // true = multilingual

mWhisper.setListener(new IWhisperListener() {
    @Override
    public void onUpdateReceived(String message) {
        // status updates
    }

    @Override
    public void onResultReceived(String result) {
        // transcription result
    }
});
```

### Transcription

Audio must be 16kHz, mono, 16-bit WAV.

```java
mWhisper.setFilePath(waveFilePath);
mWhisper.setAction(Whisper.ACTION_TRANSCRIBE);
mWhisper.start();

// later...
mWhisper.stop();
```

### Recorder

```java
Recorder mRecorder = new Recorder(this);

mRecorder.setListener(new IRecorderListener() {
    @Override
    public void onUpdateReceived(String message) { }

    @Override
    public void onDataReceived(float[] samples) {
        // optionally forward to Whisper for live recognition
        // mWhisper.writeBuffer(samples);
    }
});

mRecorder.setFilePath(waveFilePath);
mRecorder.start();

// later...
mRecorder.stop();
```

## Release Signing

The keystore and `keystore.properties` are excluded from version control. To build a signed release:

1. Create `keystore.properties` in the `whisper_android/` root:
   ```
   storeFile=<path-to-keystore>
   storePassword=<your-store-password>
   keyAlias=<your-key-alias>
   keyPassword=<your-key-password>
   ```
2. Run `./gradlew bundleRelease` (AAB) or `./gradlew assembleRelease` (APK)

Never commit `keystore.properties` or `.keystore` files to version control.

## Install via ADB

Make sure ADB is installed and USB debugging is enabled on your device.

```bash
adb install app/build/outputs/apk/release/app-release.apk
```

If you get a signature mismatch from a previous install:
```bash
adb uninstall com.whisper.android.tflitecpp
adb install app/build/outputs/apk/release/app-release.apk
```
