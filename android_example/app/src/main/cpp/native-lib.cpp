#include <jni.h>
#include <string>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <cstdio>
#include <android/log.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "whisper.h"
#include "input_features.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#define INFERENCE_ON_AUDIO_FILE 1

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

extern "C" JNIEXPORT jint JNICALL
Java_com_whisper_android_tflitecpp_MainActivity_freeModelJNI(
        JNIEnv *env,
        jobject /* this */) {
    if (g_whisper_tflite_params.buffer) {
        __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR",
                            "%s: free buffer %x memory\n", __func__,
                            g_whisper_tflite_params.buffer);
        free(g_whisper_tflite_params.buffer);

    }
    return 0;
}


// Example: load a tflite model using TF Lite C++ API
// Credit to https://github.com/ValYouW/crossplatform-tflite-object-detecion
// Credit to https://github.com/cuongvng/TF-Lite-Cpp-API-for-Android
extern "C" JNIEXPORT jstring JNICALL
Java_com_whisper_android_tflitecpp_MainActivity_loadModelJNI(
        JNIEnv *env,
        jobject /* this */,
        jobject assetManager,
        jstring fileName,
        jint is_recorded) {

    //Load Whisper Model into buffer
    jstring result = nullptr;
    struct timeval start_time, end_time;
    if (!g_whisper_tflite_params.is_whisper_tflite_initialized) {
        gettimeofday(&start_time, nullptr);
        const char *modelpath = "whisper.tflite";
        if (!(env->IsSameObject(assetManager, nullptr))) {
            AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
            AAsset *asset = AAssetManager_open(mgr, modelpath, AASSET_MODE_UNKNOWN);
            assert(asset != nullptr);

            g_whisper_tflite_params.size = AAsset_getLength(asset);
            g_whisper_tflite_params.buffer = (char *) malloc(
                    sizeof(char) * g_whisper_tflite_params.size);
            AAsset_read(asset, g_whisper_tflite_params.buffer, g_whisper_tflite_params.size);
            AAsset_close(asset);
        }

        //Load filters and vocab data from preg enerated filters_vocab_gen.bin file
        const char *vocab_filename = "filters_vocab_gen.bin";


        if (!(env->IsSameObject(assetManager, nullptr))) {
            AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
            AAsset *asset = AAssetManager_open(mgr, vocab_filename, AASSET_MODE_UNKNOWN);
            assert(asset != nullptr);
            uint32_t magic = 0;
            AAsset_read(asset, &magic, sizeof(magic));
            //@magic:USEN
            if (magic != 0x5553454e) {
                // printf("%s: invalid vocab file '%s' (bad magic)\n", __func__, fname.c_str());
                __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR",
                                    "%s: invalid vocab file '%s' (bad magic)\n", __func__,
                                    vocab_filename);
                return result;
            }
            // load mel filters
            {
                AAsset_read(asset, (char *) &filters.n_mel, sizeof(filters.n_mel));
                AAsset_read(asset, (char *) &filters.n_fft, sizeof(filters.n_fft));
                __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR", "%s: n_mel:%d n_fft:%d\n",
                                    __func__, filters.n_mel, filters.n_fft);
                filters.data.resize(filters.n_mel * filters.n_fft);
                AAsset_read(asset, (char *) filters.data.data(),
                            filters.data.size() * sizeof(float));
            }

            int32_t n_vocab = 0;
            std::string word;
            // load vocab
            {
                AAsset_read(asset, (char *) &n_vocab, sizeof(n_vocab));
                g_vocab.n_vocab = n_vocab;
                __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR", "\nn_vocab:%d\n",
                                    (int) n_vocab);

                for (int i = 0; i < n_vocab; i++) {
                    uint32_t len;
                    AAsset_read(asset, (char *) &len, sizeof(len));

                    word.resize(len);
                    AAsset_read(asset, (char *) word.data(), len);
                    g_vocab.id_to_token[i] = word;
                    //printf("len:%d",(int)len);
                    //printf("'%s'\n", g_vocab.id_to_token[i].c_str());
                }

                g_vocab.n_vocab = 51864;//add additional vocab ids
                if (g_vocab.is_multilingual()) {
                    g_vocab.token_eot++;
                    g_vocab.token_sot++;
                    g_vocab.token_prev++;
                    g_vocab.token_solm++;
                    g_vocab.token_not++;
                    g_vocab.token_beg++;
                }
                for (int i = n_vocab; i < g_vocab.n_vocab; i++) {
                    if (i > g_vocab.token_beg) {
                        word = "[_TT_" + std::to_string(i - g_vocab.token_beg) + "]";
                    } else if (i == g_vocab.token_eot) {
                        word = "[_EOT_]";
                    } else if (i == g_vocab.token_sot) {
                        word = "[_SOT_]";
                    } else if (i == g_vocab.token_prev) {
                        word = "[_PREV_]";
                    } else if (i == g_vocab.token_not) {
                        word = "[_NOT_]";
                    } else if (i == g_vocab.token_beg) {
                        word = "[_BEG_]";
                    } else {
                        word = "[_extra_token_" + std::to_string(i) + "]";
                    }
                    g_vocab.id_to_token[i] = word;
                    // printf("%s: g_vocab[%d] = '%s'\n", __func__, i, word.c_str());
                }
            }

            AAsset_close(asset);
        }


        gettimeofday(&end_time, nullptr);
        __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR",
                            "JNI mel filter extraction time %ld seconds \n",
                            (end_time.tv_sec - start_time.tv_sec));
    }
    gettimeofday(&start_time, nullptr);
    //Generate input_features for Audio file
    const char *pcmfilename = env->GetStringUTFChars(fileName, 0);// WAV input
//end of audio file processing

    gettimeofday(&end_time, nullptr);
    std::vector<float> pcmf32;
    {
        drwav wav;
        size_t audio_dataSize = 0;
        char *audio_buffer = nullptr;

        if (is_recorded) {
            if (!drwav_init_file(&wav,
                                 pcmfilename,
                                 nullptr)) {
                __android_log_print(ANDROID_LOG_VERBOSE, "Niranjan",
                                    "failed to open WAV file '%s' - check your input\n",
                                    pcmfilename);
                return result;
            }
        } else {
            if (!(env->IsSameObject(assetManager, nullptr))) {
                AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
                AAsset *asset = AAssetManager_open(mgr, pcmfilename, AASSET_MODE_UNKNOWN);
                assert(asset != nullptr);

                audio_dataSize = AAsset_getLength(asset);
                audio_buffer = (char *) malloc(sizeof(char) * audio_dataSize);
                AAsset_read(asset, audio_buffer, audio_dataSize);
                AAsset_close(asset);
            }

            if (!drwav_init_memory(&wav, audio_buffer, audio_dataSize, nullptr)) {
                __android_log_print(ANDROID_LOG_VERBOSE, "Niranjan",
                                    "failed to open WAV file '%s' - check your input\n",
                                    pcmfilename);
                return result;
            }
        }
        if (wav.channels != 1 && wav.channels != 2) {
            __android_log_print(ANDROID_LOG_VERBOSE, "Niranjan",
                                "WAV file '%s' must be mono or stereo\n", pcmfilename);

            return result;
        }

        if (wav.sampleRate != WHISPER_SAMPLE_RATE) {
            __android_log_print(ANDROID_LOG_VERBOSE, "Niranjan",
                                "WWAV file '%s' must be 16 kHz\n", pcmfilename);
            return result;
        }

        if (wav.bitsPerSample != 16) {
            __android_log_print(ANDROID_LOG_VERBOSE, "Niranjan",
                                "WAV file '%s' must be 16-bit\n", pcmfilename);
            return result;
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
    }//Hack if the audio file size is less than 30ms append with 0's
    __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR",
                        "JNI (Spectrogram)input feature extraction time %ld seconds \n",
                        (end_time.tv_sec - start_time.tv_sec));
    pcmf32.resize((WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE), 0);
    const auto processor_count = std::thread::hardware_concurrency();
    __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR", "\ncpu_cores%d\n", processor_count);
    if (!log_mel_spectrogram(pcmf32.data(), pcmf32.size(), WHISPER_SAMPLE_RATE, WHISPER_N_FFT,
                             WHISPER_HOP_LENGTH, WHISPER_N_MEL, processor_count, filters,
                             mel)) {
        fprintf(stderr, "%s: failed to compute mel spectrogram\n", __func__);
        return result;
    }
    __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR", "\nmel.n_len%d\n", mel.n_len);
    __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR", "\nmel.n_mel:%d\n", mel.n_mel);

    if (!g_whisper_tflite_params.is_whisper_tflite_initialized) {
        // Load tflite model buffer
        g_whisper_tflite_params.model =
                tflite::FlatBufferModel::BuildFromBuffer(g_whisper_tflite_params.buffer,
                                                         g_whisper_tflite_params.size);
        TFLITE_MINIMAL_CHECK(g_whisper_tflite_params.model != nullptr);

        // Build the interpreter with the InterpreterBuilder.
        // Note: all Interpreters should be built with the InterpreterBuilder,
        // which allocates memory for the Interpreter and does various set up
        // tasks so that the Interpreter can read the provided model.

        tflite::InterpreterBuilder builder(*(g_whisper_tflite_params.model),
                                           g_whisper_tflite_params.resolver);

        builder(&(g_whisper_tflite_params.interpreter));
        TFLITE_MINIMAL_CHECK(g_whisper_tflite_params.interpreter != nullptr);

        // NEW: Prepare GPU delegate.
        //  auto* delegate = TfLiteGpuDelegateV2Create(nullptr);
        // if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        //     __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR", "gpu delegate failed \n");
        // }

        // Allocate tensor buffers.
        TFLITE_MINIMAL_CHECK(g_whisper_tflite_params.interpreter->AllocateTensors() == kTfLiteOk);

        g_whisper_tflite_params.input = g_whisper_tflite_params.interpreter->typed_input_tensor<float>(
                0);
        g_whisper_tflite_params.is_whisper_tflite_initialized = true;
    }
    gettimeofday(&start_time, nullptr);
    memcpy(g_whisper_tflite_params.input, mel.data.data(), mel.n_mel * mel.n_len * sizeof(float));
    gettimeofday(&end_time, nullptr);
    __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR", "JNI input copy time %ld seconds \n",
                        (end_time.tv_sec - start_time.tv_sec));
    gettimeofday(&start_time, nullptr);
    // Run inference
    //TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    // Run inference.
    //WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
    if (g_whisper_tflite_params.interpreter->Invoke() != kTfLiteOk) return result;
    //ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

    gettimeofday(&end_time, nullptr);

    __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR", "JNI Inference time %ld seconds \n",
                        (end_time.tv_sec - start_time.tv_sec));
    int output = g_whisper_tflite_params.interpreter->outputs()[0];
    TfLiteTensor *output_tensor = g_whisper_tflite_params.interpreter->tensor(output);
    TfLiteIntArray *output_dims = output_tensor->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size = output_dims->data[output_dims->size - 1];
    //printf("output size:%d\n",output_size );
    int *output_int = g_whisper_tflite_params.interpreter->typed_output_tensor<int>(0);
    std::string text = "";
    std::string word_add;
    for (int i = 0; i < output_size; i++) {
        //printf("%d\t",output_int[i]);
        if (output_int[i] == g_vocab.token_eot) {
            break;
        }
        if ((output_int[i] != 50257) && (output_int[i] != 50362))
            text += whisper_token_to_str(output_int[i]);
    }
    __android_log_print(ANDROID_LOG_VERBOSE, "Whisper ASR", "\n%s\n", text.c_str());
    printf("\n");
    //std::string status = "Load TF Lite model successfully!";
    //free(buffer);
    return env->NewStringUTF(text.c_str());
}