---
license: mit
language:
  - en
  - fr
  - de
  - es
  - hi
  - zh
  - ja
  - ko
  - ar
  - ru
  - pt
  - it
tags:
  - whisper
  - tflite
  - speech-recognition
  - on-device
  - android
library_name: transformers
pipeline_tag: automatic-speech-recognition
---

# Whisper TFLite Model Generation and Test

Converts OpenAI Whisper speech recognition models to TFLite format for on-device inference (e.g. Android), and generates the mel filter + vocab binary file needed by native C++ runtimes.


## Requirements

- Python 3.9
- macOS, Linux, or Google Colab

Dependencies are installed automatically on first run.

## Three Generation Modes

The script supports three modes depending on the `--language` argument:

### 1. English-only (`.en` models)

```bash
python3.9 whisper_tflite_model_generation_and_test.py --model whisper-tiny.en
```

- Output: `whisper-tiny.en.tflite`
- Signature: `serving_default` (transcribe English)
- `forced_decoder_ids`: `[[2, 50359], [3, 50363]]`

### 2. Single-language (explicit language code)

```bash
python3.9 whisper_tflite_model_generation_and_test.py --model whisper-base --language fr
python3.9 whisper_tflite_model_generation_and_test.py --model whisper-base --language de
```

- Output: `whisper-base.fr.tflite`, `whisper-base.de.tflite`
- Signature: `serving_default` (transcribe the specified language)
- `forced_decoder_ids`: `[[1, <lang_token>], [2, 50359], [3, 50363]]`

### 3. Transcribe-translate (auto language detection)

```bash
python3.9 whisper_tflite_model_generation_and_test.py --model whisper-base --language auto
```

- Output: `whisper-base-transcribe-translate.tflite`
- Signatures: `serving_default` (= transcribe), `serving_transcribe`, `serving_translate`
- No language token forced — Whisper auto-detects the spoken language
- `forced_decoder_ids (transcribe)`: `[[2, 50359], [3, 50363]]`
- `forced_decoder_ids (translate)`: `[[2, 50358], [3, 50363]]`

## Usage

```bash
# Default: whisper-tiny.en (English-only)
python3.9 whisper_tflite_model_generation_and_test.py

# Single-language French model
python3.9 whisper_tflite_model_generation_and_test.py --model whisper-base --language fr

# Transcribe-translate model (auto-detect language)
python3.9 whisper_tflite_model_generation_and_test.py --model whisper-base --language auto

# Test translate signature specifically
python3.9 whisper_tflite_model_generation_and_test.py --model whisper-small --language auto --task translate

# Show all options
python3.9 whisper_tflite_model_generation_and_test.py --help
```

## Command Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `whisper-tiny.en` | Whisper model to convert |
| `--language` | `en` | Language code (`en`, `fr`, `de`, ...) or `auto` for auto-detection |
| `--task` | `transcribe` | `transcribe` or `translate`. Only relevant with `--language auto` |
| `--multilingual` / `--no-multilingual` | `--multilingual` | Vocab binary type. Auto-set to `--no-multilingual` for `.en` models |

### Supported Models

| Model | Type | Parameters | Required VRAM | Relative Speed |
|---|---|---|---|---|
| `whisper-tiny.en` | English-only | ~39M | ~1 GB | ~10x |
| `whisper-tiny` | Multilingual | ~39M | ~1 GB | ~10x |
| `whisper-base.en` | English-only | ~74M | ~1 GB | ~7x |
| `whisper-base` | Multilingual | ~74M | ~1 GB | ~7x |
| `whisper-small.en` | English-only | ~244M | ~2 GB | ~4x |
| `whisper-small` | Multilingual | ~244M | ~2 GB | ~4x |
| `whisper-medium.en` | English-only | ~769M | ~5 GB | ~2x |
| `whisper-medium` | Multilingual | ~769M | ~5 GB | ~2x |
| `whisper-large` | Multilingual | ~1550M | ~10 GB | 1x |
| `whisper-large-v3` | Multilingual | ~1550M | ~10 GB | 1x |
| `whisper-turbo` | Multilingual | ~809M | ~6 GB | ~8x |

### Supported Languages

`en`, `fr`, `hi`, `ko`, `de`, `zh`, `ja`, `es`, `ar`, `ru`, `pt`, `it`, `nl`, `sv`, `pl`, `da`, `fi`, and many more.

Use `auto` for language auto-detection. See the [full list](https://github.com/openai/whisper).

## TFLite Serving Signatures

| Mode | Signatures | Description |
|---|---|---|
| English-only (`.en`) | `serving_default` | Transcribe English |
| Single-language | `serving_default` | Transcribe the forced language |
| Transcribe-translate (`auto`) | `serving_default`, `serving_transcribe`, `serving_translate` | Auto-detect language, transcribe or translate to English |

Token reference:
- `50358` = `<|translate|>`
- `50359` = `<|transcribe|>`
- `50363` = `<|notimestamps|>`

## What the Script Does

| Step | Description |
|---|---|
| 0 | Install/verify Python dependencies |
| 1 | Configure model parameters and fetch decoder token mappings |
| 2 | Load the Whisper model and run a test transcription (English: LibriSpeech, other languages: [Google FLEURS](https://huggingface.co/datasets/google/fleurs)) |
| 3 | Patch `TFForceTokensLogitsProcessor` to avoid NaN values during TFLite export |
| 4 | Wrap the model with serving signature(s) and save as TF SavedModel |
| 5 | Convert the SavedModel to TFLite with dynamic range quantization |
| 6 | Verify the TFLite model produces correct output via the TFLite Interpreter |
| 7 | (Optional) Test the TFLite model against `.wav` audio files |
| 8 | Generate the mel filters + vocab binary file |

## Output Files

```
# English-only
whisper-tiny.en.tflite
filters_vocab_en.bin

# Single-language (e.g. French)
whisper-base.fr.tflite
filters_vocab_multilingual.bin

# Transcribe-translate (auto)
whisper-base-transcribe-translate.tflite
filters_vocab_multilingual.bin
```

These files are what you need for on-device Whisper inference on Android or other embedded platforms.


## Citing

If you are using the Whisper tflite model, please cite:

```bibtex
@misc{nyadla-sys,
  author={Niranjan Yadla},
  title={{Whisper TFLite: OpenAI Whisper Model Port for Edge Devices}},
  year=2022,
  howpublished={GitHub Repository},
  url={https://github.com/nyadla-sys/whisper.tflite},
  url={https://github.com/moonshine-ai/openai-whisper},
  note={Original TFLite implementation of OpenAI Whisper for on-device automatic speech recognition}
}
```
