# Whisper Android — Offline Speech Recognition

Offline speech recognition for Android using OpenAI Whisper quantized TFLite models.

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

5. Model and vocab files are already bundled in `app/src/main/assets/`:
   | Model file | Vocab file | Language |
   |---|---|---|
   | `whisper-tiny-en.tflite` | `filters_vocab_en.bin` | English only |
   | `whisper-tiny.tflite` | `filters_vocab_multilingual.bin` | Multilingual |

6. Build a release APK (signed with the release keystore):
   ```bash
   ./gradlew assembleRelease
   ```
   Output: `app/build/outputs/apk/release/app-release.apk`

   Or build a debug APK:
   ```bash
   ./gradlew assembleDebug
   ```
   Or use the Run button in Android Studio.

## Running Tests

JVM property-based tests (jqwik 1.8.4):
```bash
./gradlew test
```

## Models

Use the English-only model for better performance when you only need English transcription.
Use the multilingual model when transcribing other languages.

## Whisper API Integration

### Initialization

```java
Whisper mWhisper = new Whisper(this);

// English only
String modelPath = getFilePath("whisper-tiny-en.tflite");
String vocabPath = getFilePath("filters_vocab_en.bin");
mWhisper.loadModel(modelPath, vocabPath, false); // false = English only

// Multilingual
// String modelPath = getFilePath("whisper-tiny.tflite");
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

The keystore and `keystore.properties` are excluded from version control. To build a signed release APK:

1. The keystore is at `app/whisper-release.keystore` (generated once, keep it safe)
2. `keystore.properties` in the `whisper_android/` root holds the credentials — create it if missing:
   ```
   storeFile=whisper-release.keystore
   storePassword=<your-store-password>
   keyAlias=whisper-release
   keyPassword=<your-key-password>
   ```
3. Run `./gradlew assembleRelease`

Never commit `keystore.properties` or the `.keystore` file to version control.

## Sideload the Pre-built APK

A pre-built APK is available in the `pre_built_apk/` folder if you want to try the app without building from source.

### Enable Unknown Sources on your Android device

1. Go to Settings → Apps (or Privacy/Security depending on your device)
2. Tap "Install unknown apps" or "Special app access"
3. Select the app you'll use to install (e.g. Files, Chrome) and toggle "Allow from this source"

### Install via ADB (recommended)

Make sure ADB is installed and USB debugging is enabled on your device (Settings → Developer Options → USB Debugging).

```bash
adb install whisper_android/pre_built_apk/WhisperASR.apk
```

If you get a signature mismatch from a previous install:
```bash
adb uninstall com.whispertflite
adb install whisper_android/pre_built_apk/WhisperASR.apk
```

### Install via file transfer

1. Copy `pre_built_apk/WhisperASR.apk` to your Android device (USB, Google Drive, email, etc.)
2. Open a file manager on the device and tap the APK
3. Tap "Install" when prompted

### Permissions required

The app will request microphone permission on first launch. Grant it to enable recording and transcription.

## Demo

[![Demo Video](https://img.youtube.com/vi/w9pohi9NQrg/0.jpg)](https://www.youtube.com/watch?v=w9pohi9NQrg)
