#!/usr/bin/env python3
"""
Whisper TFLite Model Generation and Test

Converts an OpenAI Whisper model to TFLite format and generates the mel
filter + vocab binary file needed by native C++ runtimes.


Prerequisites:
    Python 3.9
    Dependencies are installed automatically on first run.

Usage:
    python3.9 whisper_tflite_model_generation_and_test.py
    python3.9 whisper_tflite_model_generation_and_test.py --model whisper-base --language fr
    python3.9 whisper_tflite_model_generation_and_test.py --model whisper-base --language auto
"""

import argparse
import os
import subprocess
import sys

# ============================================================================
# Parse command line arguments
# ============================================================================
parser = argparse.ArgumentParser(
    prog="python3.9 whisper_tflite_model_generation_and_test.py",
    description="Convert OpenAI Whisper model to TFLite and generate vocab binary.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # English-only model (single serving_default signature)
  python3.9 whisper_tflite_model_generation_and_test.py
  python3.9 whisper_tflite_model_generation_and_test.py --model whisper-base.en

  # Single-language model (serving_default forces that language)
  python3.9 whisper_tflite_model_generation_and_test.py --model whisper-base --language fr
  python3.9 whisper_tflite_model_generation_and_test.py --model whisper-base --language de

  # Transcribe-translate model (auto-detect language, dual signatures)
  python3.9 whisper_tflite_model_generation_and_test.py --model whisper-base --language auto
  python3.9 whisper_tflite_model_generation_and_test.py --model whisper-small --language auto --task translate

Output naming convention:
  .en model          -> whisper-tiny.en.tflite
  single language    -> whisper-base.fr.tflite
  auto (transcribe-translate) -> whisper-base-transcribe-translate.tflite

Supported models:
  whisper-tiny.en    Tiny English-only model (~39M params)
  whisper-tiny       Tiny multilingual model (~39M params)
  whisper-base.en    Base English-only model (~74M params)
  whisper-base       Base multilingual model (~74M params)
  whisper-small.en   Small English-only model (~244M params)
  whisper-small      Small multilingual model (~244M params)

Supported languages:
  en, fr, hi, ko, de, zh, ja, es, ar, ru, pt, it, nl, sv, pl, da, fi,
  and many more. Use 'auto' for language auto-detection.
  Full list: https://github.com/openai/whisper
""",
)
parser.add_argument(
    "--model",
    type=str,
    default="whisper-tiny.en",
    choices=[
        "whisper-tiny.en", "whisper-tiny",
        "whisper-base.en", "whisper-base",
        "whisper-small.en", "whisper-small",
    ],
    help="Whisper model name (default: whisper-tiny.en)",
)
parser.add_argument(
    "--language",
    type=str,
    default="en",
    help="Language code (en, fr, de, ...) or 'auto' for auto-detection (default: en)",
)
parser.add_argument(
    "--task",
    type=str,
    default="transcribe",
    choices=["transcribe", "translate"],
    help="Task for testing: transcribe or translate (default: transcribe). Only relevant with --language auto.",
)
parser.add_argument(
    "--multilingual",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Vocab binary type (default: True). Auto-set to False for .en models.",
)
args = parser.parse_args()

# Auto-detect settings for .en (English-only) models
is_english_only = args.model.endswith(".en")
if is_english_only:
    args.language = "en"
    args.multilingual = False

is_auto_language = args.language == "auto"

# ============================================================================
# Step 0: Install dependencies
# ============================================================================
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.14.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.33.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets==2.14.0", "pyarrow==14.0.2", "fsspec==2023.6.0", "soundfile", "librosa"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-whisper"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

import requests
import json
import numpy as np
import tensorflow as tf
import transformers
import datasets

from typing import List
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperFeatureExtractor,
    TFWhisperForConditionalGeneration,
    WhisperTokenizer,
    TFForceTokensLogitsProcessor,
    TFLogitsProcessor,
)

# Whisper supported languages (from openai/whisper tokenizer.py)
WHISPER_LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}

# Validate language code
if args.language != "auto" and args.language not in WHISPER_LANGUAGES:
    print(f"Error: Unknown language code '{args.language}'.")
    print(f"Supported: {', '.join(WHISPER_LANGUAGES.keys())}")
    sys.exit(1)

lang_display = (
    f"{args.language} ({WHISPER_LANGUAGES[args.language].title()})"
    if args.language in WHISPER_LANGUAGES else args.language
)

# ============================================================================
# Step 1: Configure model parameters
# ============================================================================
print("=" * 60)
print("Step 1: Configuring model parameters")
print("=" * 60)
print(f"  Model:        {args.model}")
print(f"  Language:      {lang_display}")
print(f"  Task:          {args.task}")
print(f"  Multilingual:  {args.multilingual}")

model_name = args.model
pretrained_model = f"openai/{model_name}"

# Determine output file naming convention
if is_english_only:
    # whisper-tiny.en.tflite
    tflite_model_path = f"{model_name}.tflite"
    saved_model_dir = f"tf_{model_name}_saved"
elif is_auto_language:
    # whisper-base-transcribe-translate.tflite
    tflite_model_path = f"{model_name}-transcribe-translate.tflite"
    saved_model_dir = f"tf_{model_name}_transcribe_translate_saved"
elif args.language == "en":
    # whisper-small.tflite (multilingual model forced to English)
    tflite_model_path = f"{model_name}.tflite"
    saved_model_dir = f"tf_{model_name}_saved"
else:
    # whisper-base.fr.tflite
    tflite_model_path = f"{model_name}.{args.language}.tflite"
    saved_model_dir = f"tf_{model_name}_{args.language}_saved"

print(f"  TFLite output:  {tflite_model_path}")
print(f"  SavedModel dir: {saved_model_dir}")

# Fetch token code mappings from HuggingFace
url = "https://huggingface.co/openai/whisper-large/resolve/main/added_tokens.json"
response = requests.get(url)
if response.status_code == 200:
    code_mappings = response.json()
else:
    print(f"Failed to download token mappings. Status code: {response.status_code}")
    sys.exit(1)

transcribe_code = code_mappings["<|transcribe|>"]
translate_code = code_mappings["<|translate|>"]
notimestamps_code = code_mappings["<|notimestamps|>"]

# Build forced_decoder_ids based on mode
if is_english_only:
    # .en models: transcribe only, no language token
    forced_decoder_ids = [
        [2, transcribe_code],
        [3, notimestamps_code],
    ]
    print(f"  Mode: English-only")
    print(f"  forced_decoder_ids: {forced_decoder_ids}")

elif is_auto_language:
    # Auto-detect language: no language token, separate transcribe/translate
    forced_decoder_ids_transcribe = [
        [2, transcribe_code],
        [3, notimestamps_code],
    ]
    forced_decoder_ids_translate = [
        [2, translate_code],
        [3, notimestamps_code],
    ]
    forced_decoder_ids = (
        forced_decoder_ids_transcribe if args.task == "transcribe"
        else forced_decoder_ids_translate
    )
    print(f"  Mode: Transcribe-translate (auto language detection)")
    print(f"  forced_decoder_ids (transcribe): {forced_decoder_ids_transcribe}")
    print(f"  forced_decoder_ids (translate):  {forced_decoder_ids_translate}")

else:
    # Single language: force specific language token
    language_code = f"<|{args.language}|>"
    language_token = code_mappings[language_code]
    forced_decoder_ids = [
        [1, language_token],
        [2, transcribe_code],
        [3, notimestamps_code],
    ]
    print(f"  Mode: Single-language ({lang_display})")
    print(f"  forced_decoder_ids: {forced_decoder_ids}")

# ============================================================================
# Step 2: Load the model and run initial inference
# ============================================================================
print("\n" + "=" * 60)
print("Step 2: Loading model and running initial inference")
print("=" * 60)

feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model)
tokenizer = WhisperTokenizer.from_pretrained(pretrained_model, predict_timestamps=True)
processor = WhisperProcessor(feature_extractor, tokenizer)
model = TFWhisperForConditionalGeneration.from_pretrained(pretrained_model)

# Load test dataset matching the language
test_lang = "en" if is_auto_language else args.language
if test_lang == "en":
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    test_audio = ds[0]["audio"]["array"]
    test_sr = ds[0]["audio"]["sampling_rate"]
else:
    # Use FLEURS for multilingual test samples (public, no login required)
    # FLEURS uses full language names with underscores, e.g. "fr_fr", "de_de"
    fleurs_lang_map = {
        "af": "af_za", "am": "am_et", "ar": "ar_eg", "as": "as_in", "az": "az_az",
        "be": "be_by", "bg": "bg_bg", "bn": "bn_in", "bs": "bs_ba", "ca": "ca_es",
        "cs": "cs_cz", "cy": "cy_gb", "da": "da_dk", "de": "de_de", "el": "el_gr",
        "en": "en_us", "es": "es_419", "et": "et_ee", "fa": "fa_ir", "fi": "fi_fi",
        "fr": "fr_fr", "gl": "gl_es", "gu": "gu_in", "ha": "ha_ng", "he": "he_il",
        "hi": "hi_in", "hr": "hr_hr", "hu": "hu_hu", "hy": "hy_am", "id": "id_id",
        "is": "is_is", "it": "it_it", "ja": "ja_jp", "ka": "ka_ge", "kk": "kk_kz",
        "km": "km_kh", "kn": "kn_in", "ko": "ko_kr", "lo": "lo_la", "lt": "lt_lt",
        "lv": "lv_lv", "mk": "mk_mk", "ml": "ml_in", "mn": "mn_mn", "mr": "mr_in",
        "ms": "ms_my", "my": "my_mm", "ne": "ne_np", "nl": "nl_nl", "no": "nb_no",
        "pa": "pa_in", "pl": "pl_pl", "pt": "pt_br", "ro": "ro_ro", "ru": "ru_ru",
        "sk": "sk_sk", "sl": "sl_si", "sr": "sr_rs", "sv": "sv_se", "sw": "sw_ke",
        "ta": "ta_in", "te": "te_in", "th": "th_th", "tr": "tr_tr", "uk": "uk_ua",
        "ur": "ur_pk", "uz": "uz_uz", "vi": "vi_vn", "zh": "cmn_hans_cn",
    }
    fleurs_code = fleurs_lang_map.get(test_lang)
    if fleurs_code:
        print(f"  Loading FLEURS test sample for '{test_lang}' ({fleurs_code})...")
        ds = load_dataset(
            "google/fleurs",
            fleurs_code,
            split="test",
            streaming=True,
        )
        sample = next(iter(ds))
        test_audio = sample["audio"]["array"]
        test_sr = sample["audio"]["sampling_rate"]
    else:
        print(f"  No public test dataset found for '{test_lang}'. Using English sample.")
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        test_audio = ds[0]["audio"]["array"]
        test_sr = ds[0]["audio"]["sampling_rate"]

inputs = feature_extractor(test_audio, sampling_rate=test_sr, return_tensors="tf")
input_features = inputs.input_features

generated_ids = model.generate(input_features=input_features)
print(generated_ids)
transcription = processor.tokenizer.decode(generated_ids[0])
print(f"Initial transcription: {transcription}")

# ============================================================================
# Step 3: Patch TFForceTokensLogitsProcessor for TFLite export
# ============================================================================
print("\n" + "=" * 60)
print("Step 3: Applying TFForceTokensLogitsProcessor patch")
print("=" * 60)


def my__init__(self, force_token_map: List[List[int]]):
    force_token_map = dict(force_token_map)
    force_token_array = np.ones((max(force_token_map.keys()) + 1), dtype=np.int32) * -1
    for index, token in force_token_map.items():
        if token is not None:
            force_token_array[index] = token
    self.force_token_array = tf.convert_to_tensor(force_token_array, dtype=tf.int32)


def my__call__(
    self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int
) -> tf.Tensor:
    def _force_token(generation_idx):
        batch_size = scores.shape[0]
        current_token = self.force_token_array[generation_idx]
        new_scores = tf.ones_like(scores, dtype=scores.dtype) * -float(1)
        indices = tf.stack(
            (tf.range(batch_size), tf.tile([current_token], [batch_size])), axis=1
        )
        updates = tf.zeros((batch_size,), dtype=scores.dtype)
        new_scores = tf.tensor_scatter_nd_update(new_scores, indices, updates)
        return new_scores

    scores = tf.cond(
        tf.greater_equal(cur_len, tf.shape(self.force_token_array)[0]),
        lambda: tf.identity(scores),
        lambda: tf.cond(
            tf.greater_equal(self.force_token_array[cur_len], 0),
            lambda: _force_token(cur_len),
            lambda: scores,
        ),
    )
    return scores


TFForceTokensLogitsProcessor.__init__ = my__init__
TFForceTokensLogitsProcessor.__call__ = my__call__
print("Patch applied successfully.")


# ============================================================================
# Step 4: Define serving signatures and save as TF SavedModel
# ============================================================================
print("\n" + "=" * 60)
print("Step 4: Saving model in TF SavedModel format")
print("=" * 60)

if is_english_only or (not is_auto_language):
    # English-only OR single-language: one serving_default signature
    class GenerateModel(tf.Module):
        def __init__(self, model):
            super(GenerateModel, self).__init__()
            self.model = model

        @tf.function(
            input_signature=[
                tf.TensorSpec((1, 80, 3000), tf.float32, name="input_features"),
            ],
        )
        def serving(self, input_features):
            outputs = self.model.generate(
                input_features,
                max_new_tokens=448,
                return_dict_in_generate=True,
                forced_decoder_ids=forced_decoder_ids,
            )
            return {"sequences": outputs["sequences"]}

    generate_model = GenerateModel(model=model)
    tf.saved_model.save(
        generate_model,
        saved_model_dir,
        signatures={"serving_default": generate_model.serving},
    )
    print("Exported signature: serving_default")

else:
    # Auto language: transcribe + translate + serving_default (= transcribe)
    class GenerateModel(tf.Module):
        def __init__(self, model):
            super(GenerateModel, self).__init__()
            self.model = model

        @tf.function(
            input_signature=[
                tf.TensorSpec((1, 80, 3000), tf.float32, name="input_features"),
            ],
        )
        def transcribe(self, input_features):
            outputs = self.model.generate(
                input_features,
                max_new_tokens=448,
                return_dict_in_generate=True,
                forced_decoder_ids=forced_decoder_ids_transcribe,
            )
            return {"sequences": outputs["sequences"]}

        @tf.function(
            input_signature=[
                tf.TensorSpec((1, 80, 3000), tf.float32, name="input_features"),
            ],
        )
        def translate(self, input_features):
            outputs = self.model.generate(
                input_features,
                max_new_tokens=448,
                return_dict_in_generate=True,
                forced_decoder_ids=forced_decoder_ids_translate,
            )
            return {"sequences": outputs["sequences"]}

    generate_model = GenerateModel(model=model)
    tf.saved_model.save(
        generate_model,
        saved_model_dir,
        signatures={
            "serving_default": generate_model.transcribe,
            "serving_transcribe": generate_model.transcribe,
            "serving_translate": generate_model.translate,
        },
    )
    print("Exported signatures: serving_default, serving_transcribe, serving_translate")

print(f"SavedModel saved to: {saved_model_dir}")

# ============================================================================
# Step 5: Convert to TFLite
# ============================================================================
print("\n" + "=" * 60)
print("Step 5: Converting to TFLite format")
print("=" * 60)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {tflite_model_path}")

# ============================================================================
# Step 6: Test TFLite model
# ============================================================================
print("\n" + "=" * 60)
print("Step 6: Testing TFLite model with dataset sample")
print("=" * 60)

interpreter = tf.lite.Interpreter(tflite_model_path)

if is_auto_language:
    sig_name = f"serving_{args.task}"
else:
    sig_name = "serving_default"

tflite_generate = interpreter.get_signature_runner(sig_name)
print(f"Testing with signature: {sig_name}")

generated_ids = tflite_generate(input_features=input_features)["sequences"]
transcription = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(f"TFLite transcription: {transcription}")


# ============================================================================
# Step 7: Test with audio files (optional)
# ============================================================================
print("\n" + "=" * 60)
print("Step 7: Testing with audio files (optional)")
print("=" * 60)

try:
    from faster_whisper import decode_audio

    audio_lang = "en" if is_auto_language else args.language
    audio_folder_path = f"audio_samples/{audio_lang}"
    if not os.path.exists("audio_samples"):
        subprocess.run(
            ["git", "clone", "https://github.com/vilassn/audio_samples.git"],
            check=True,
        )

    if not os.path.isdir(audio_folder_path):
        print(f"Audio folder not found at: {audio_folder_path}")
        print(f"No audio samples for '{audio_lang}'. Skipping.")
        audio_folder_path = None

    feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model)
    tokenizer = WhisperTokenizer.from_pretrained(
        pretrained_model, predict_timestamps=True
    )
    processor = WhisperProcessor(feature_extractor, tokenizer)

    interpreter = tf.lite.Interpreter(tflite_model_path)
    tflite_generate = interpreter.get_signature_runner(sig_name)

    if audio_folder_path:
        print(f"\nTesting audio files from: {audio_folder_path}\n")
        for audio_file_name in sorted(os.listdir(audio_folder_path)):
            audio_file_path = os.path.join(audio_folder_path, audio_file_name)

            if audio_file_name.endswith(".wav"):
                print(f"Processing {audio_file_name}...")
                input_audio = decode_audio(audio_file_path, sampling_rate=16000)
                input_features = feature_extractor(
                    input_audio, sampling_rate=16000, return_tensors="tf"
                ).input_features

                generated_ids = tflite_generate(input_features=input_features)[
                    "sequences"
                ]
                transcription = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                print(f"{transcription}\n")

except ImportError:
    print("faster-whisper not installed. Skipping audio file testing.")
    print("Install with: pip install faster-whisper")

# ============================================================================
# Step 8: Generate mel filters + vocab binary file
# ============================================================================
print("\n" + "=" * 60)
print("Step 8: Generating mel filters and vocab binary file")
print("=" * 60)

import struct
import base64
import torch
from pathlib import Path

multilingual = args.multilingual

whisper_repo_dir = "whisper_repo"
if not os.path.exists(whisper_repo_dir):
    subprocess.run(
        ["git", "clone", "https://github.com/openai/whisper.git", whisper_repo_dir],
        check=True,
    )

n_mels = 80
with np.load(Path(whisper_repo_dir) / "whisper" / "assets" / "mel_filters.npz") as f:
    filters = torch.from_numpy(f[f"mel_{n_mels}"])

tokenizer_path = Path(whisper_repo_dir) / "whisper" / "assets" / (
    "multilingual.tiktoken" if multilingual else "gpt2.tiktoken"
)

with open(tokenizer_path, "rb") as f:
    contents = f.read()
    tokens = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }

fname_out = Path(".") / (
    "filters_vocab_multilingual.bin" if multilingual else "filters_vocab_en.bin"
)

with fname_out.open("wb") as fout:
    fout.write(struct.pack("i", 0x5553454E))
    fout.write(struct.pack("i", filters.shape[0]))
    fout.write(struct.pack("i", filters.shape[1]))
    for i in range(filters.shape[0]):
        for j in range(filters.shape[1]):
            fout.write(struct.pack("f", filters[i][j]))
    fout.write(struct.pack("i", len(tokens)))
    for key in tokens:
        fout.write(struct.pack("i", len(key)))
        fout.write(key)

print(f"Vocab binary file saved to: {fname_out}")

print("\nDone!")
