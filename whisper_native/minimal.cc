/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "whisper.h"
#include "input_features.h"
#include "tflt-vocab-mel.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }
std::string removeExtraSpaces(const std::string& input) {
  std::string result;
  result.reserve(input.length());
  bool space = false;

  for (char c : input) {
    if (c == ' ') {
      if (!space) {
        result += c;
      }
      space = true;
    } else {
      result += c;
      space = false;
    }
  }

  return result;
}

int main(int argc, char* argv[]) {
  if ((argc != 2) && (argc != 3)) {
    fprintf(stderr, "'minimal <tflite model>' or 'minimal <tflite model> <pcm_file name>'\n");
    return 1;
  }
  const char* filename = argv[1];
  WhisperMelSpectrogram mel; // Use the correct struct from whisper.h
  struct timeval start_time, end_time;

  // Create a pointer to the start of the unsigned char array
  unsigned char *ptr = tflt_vocab_mel_bin;
  // Read the magic number
  uint32_t magic = 0;
  memcpy(&magic, ptr, sizeof(magic));
  //tflt
  if (magic != 0x74666C74) {
    printf("Invalid vocab file (bad magic)\n");
    return 0;
  }
  ptr += sizeof(magic); // Move the pointer to the next position

  WhisperFilters filters; // Use the correct struct from whisper.h
  // Load mel filters
  memcpy(&filters.numMel, ptr, sizeof(filters.numMel));
  ptr += sizeof(filters.numMel);

  memcpy(&filters.numFFT, ptr, sizeof(filters.numFFT));
  ptr += sizeof(filters.numFFT);

  // Allocate memory for the vector and copy data
  filters.data.resize(filters.numMel * filters.numFFT);
  memcpy(filters.data.data(), ptr, filters.numMel * filters.numFFT * sizeof(float));
  ptr += filters.numMel * filters.numFFT * sizeof(float);

  // Load vocab
  int32_t n_vocab = 0;
  memcpy(&n_vocab, ptr, sizeof(n_vocab));
  ptr += sizeof(n_vocab);

  gVocab.numTokens = n_vocab;  // Update the vocabulary size based on whisper.h
  printf("\nn_vocab:%d\n", (int)n_vocab);

  char word[256]; // Assuming a maximum word length of 255 characters
  for (int i = 0; i < n_vocab; i++) {
    uint32_t len;
    memcpy(&len, ptr, sizeof(len));
    ptr += sizeof(len);

    memcpy(word, ptr, len);
    word[len] = '\0'; // Null-terminate the string
    ptr += len;

    gVocab.idToToken[i] = std::string(word);
  }
  
  // Generate input_features for Audio file
  if (argc == 3) {
    const char* pcmfilename = argv[2];
    // WAV input
    std::vector<float> pcmf32;
    {
      drwav wav;
      if (!drwav_init_file(&wav, pcmfilename, NULL)) {
        fprintf(stderr, "%s: failed to open WAV file '%s' - check your input\n", argv[0], pcmfilename);
        return 3;
      }

      if (wav.channels != 1 && wav.channels != 2) {
        fprintf(stderr, "%s: WAV file '%s' must be mono or stereo\n", argv[0], pcmfilename);
        return 4;
      }

      if (wav.sampleRate != kWhisperSampleRate) {  // Update to use the correct sample rate
        fprintf(stderr, "%s: WAV file '%s' must be 16 kHz\n", argv[0], pcmfilename);
        return 5;
      }

      if (wav.bitsPerSample != 16) {
        fprintf(stderr, "%s: WAV file '%s' must be 16-bit\n", argv[0], pcmfilename);
        return 6;
      }

      int n = wav.totalPCMFrameCount;

      std::vector<int16_t> pcm16;
      pcm16.resize(n * wav.channels);
      drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
      drwav_uninit(&wav);
      // convert to mono, float
      pcmf32.resize(n);
      if (wav.channels == 1) {
        for (int i = 0; i < n; i++) {
          pcmf32[i] = float(pcm16[i]) / 32768.0f;
        }
      } else {
        for (int i = 0; i < n; i++) {
          pcmf32[i] = float(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
        }
      }
    }

    // Hack if the audio file size is less than 30ms, append with 0's
    pcmf32.resize((kWhisperSampleRate * kWhisperChunkSize), 0);
    if (!logMelSpectrogram(pcmf32.data(), pcmf32.size(), kWhisperSampleRate, kWhisperNFFT, kWhisperHopLength, kWhisperNMEL, 1, filters, mel)) {
      fprintf(stderr, "%s: failed to compute mel spectrogram\n", __func__);
      return -1;
    }

    printf("\nmel.n_len%d\n", mel.numFrames); // Update to use the correct struct members
    printf("\nmel.n_mel:%d\n", mel.numMelFilters);    // Update to use the correct struct members
  }//end of audio file processing

  // Load tflite model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Get information about the memory area to use for the model's input.
  float* input = interpreter->typed_input_tensor<float>(0);
  if (argc == 2) {
    // Load pre-generated input_features
    memcpy(input, _content_input_features_bin, kWhisperNMEL * kWhisperMelLen * sizeof(float));
  } else if (argc == 3) {
    // Use the processed audio data as input
    memcpy(input, mel.data.data(), mel.numMelFilters * mel.numFrames * sizeof(float)); // Update to use the correct struct members
  }

  gettimeofday(&start_time, NULL);
  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  gettimeofday(&end_time, NULL);
  printf("Inference time %ld seconds \n", (end_time.tv_sec - start_time.tv_sec));

  int output = interpreter->outputs()[0];
  TfLiteTensor *output_tensor = interpreter->tensor(output);
  TfLiteIntArray *output_dims = output_tensor->dims;
  auto output_size = output_dims->data[output_dims->size - 1];
  int *output_int = interpreter->typed_output_tensor<int>(0);
  std::string text = "";

  for (int i = 0; i < output_size; i++) {
    if (output_int[i] == gVocab.tokenEOT) {
      break;
    }
    if (output_int[i] < gVocab.tokenEOT) {
      text += whisperTokenToString(output_int[i]);
    }
  }

  // Remove extra spaces between words
  text = removeExtraSpaces(text);

  printf("\n%s\n", text.c_str());
  printf("\n");

  return 0;
}
