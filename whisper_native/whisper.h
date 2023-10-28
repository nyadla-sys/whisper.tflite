// Courtesy from @ggerganov https://github.com/ggerganov/whisper.cpp
#include <iostream>
#include <fstream>
#include <thread>
#include <sys/time.h>
#include <vector>
#include <map>
#include <cmath>
#include <string>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

// Constants
constexpr int kNumGoldenGeneratedIDs = 21;
constexpr int kGoldenGeneratedIDs[kNumGoldenGeneratedIDs] = {50257, 50362, 1770, 13, 2264, 346, 353, 318, 262, 46329, 286, 262, 3504, 6097, 11, 290, 356, 389, 9675, 284, 7062};
constexpr int kWhisperSampleRate = 16000;
constexpr int kWhisperNFFT = 400;
constexpr int kWhisperNMEL = 80;
constexpr int kWhisperHopLength = 160;
constexpr int kWhisperChunkSize = 30;
constexpr int kWhisperMelLen = 3000;

// Structure for Whisper Vocab
struct WhisperVocab {
    using ID = int32_t;
    using Token = std::string;

    int numTokens = 51864;
    std::map<ID, Token> idToToken;

    ID tokenEOT = 50256;
    ID tokenSOT = 50257;
    ID tokenPrev = 50360;
    ID tokenSOLM = 50361; // ??
    ID tokenNot = 50362;  // no timestamps
    ID tokenBeg = 50363;

    // Available tasks
    static const ID tokenTransliterate = 50358;
    static const ID tokenTranscribe = 50359;

    bool isMultilingual() const {
        return numTokens == 51865;
    }
};

// Global Whisper Vocab
WhisperVocab gVocab;

// Structure for Whisper Filters
struct WhisperFilters {
    int32_t numMel;
    int32_t numFFT;
    std::vector<float> data;
};

// Structure for Mel Spectrogram
struct WhisperMelSpectrogram {
    int numFrames;
    int numMelFilters;
    std::vector<float> data;
};

// Function to print vector elements
void printVector(const std::vector<float> &vec) {
    std::cout << "Vector elements: ";
    for (const float &value : vec) {
        std::cout << value << ' ';
    }
    std::cout << '\n';
}

// Function to convert Whisper token to string
const char *whisperTokenToString(int token) {
    return gVocab.idToToken.at(token).c_str();
}

// Naive Discrete Fourier Transform
void dft(const std::vector<float> &input, std::vector<float> &output) {
    int N = input.size();
    output.resize(N * 2);

    for (int k = 0; k < N; k++) {
        float real = 0;
        float imag = 0;

        for (int n = 0; n < N; n++) {
            float angle = 2 * M_PI * k * n / N;
            real += input[n] * cos(angle);
            imag -= input[n] * sin(angle);
        }

        output[k * 2 + 0] = real;
        output[k * 2 + 1] = imag;
    }
}

// Cooley-Tukey FFT
void fft(const std::vector<float> &input, std::vector<float> &output) {
    output.resize(input.size() * 2);
    int N = input.size();

    if (N == 1) {
        output[0] = input[0];
        output[1] = 0;
        return;
    }

    if (N % 2 == 1) {
        dft(input, output);
        return;
    }

    std::vector<float> even;
    std::vector<float> odd;

    for (int i = 0; i < N; i++) {
        if (i % 2 == 0) {
            even.push_back(input[i]);
        } else {
            odd.push_back(input[i]);
        }
    }

    std::vector<float> evenFFT;
    std::vector<float> oddFFT;

    fft(even, evenFFT);
    fft(odd, oddFFT);

    for (int k = 0; k < N / 2; k++) {
        float theta = 2 * M_PI * k / N;
        float re = cos(theta);
        float im = -sin(theta);

        float reOdd = oddFFT[2 * k + 0];
        float imOdd = oddFFT[2 * k + 1];

        output[2 * k + 0] = evenFFT[2 * k + 0] + re * reOdd - im * imOdd;
        output[2 * k + 1] = evenFFT[2 * k + 1] + re * imOdd + im * reOdd;

        output[2 * (k + N / 2) + 0] = evenFFT[2 * k + 0] - re * reOdd + im * imOdd;
        output[2 * (k + N / 2) + 1] = evenFFT[2 * k + 1] - re * imOdd - im * reOdd;
    }
}

// Function to compute log mel spectrogram
bool logMelSpectrogram(const float *samples, const int numSamples, const int sampleRate,
                       const int fftSize, const int fftStep, const int numMelFilters,
                       const int numThreads, const WhisperFilters &filters,
                       WhisperMelSpectrogram &melSpectrogram) {
    // Hanning window
    std::vector<float> hann(fftSize);
    for (int i = 0; i < fftSize; i++) {
        hann[i] = 0.5 * (1.0 - cos((2.0 * M_PI * i) / fftSize));
    }

    melSpectrogram.numMelFilters = numMelFilters;
    melSpectrogram.numFrames = (numSamples) / fftStep;
    melSpectrogram.data.resize(melSpectrogram.numMelFilters * melSpectrogram.numFrames);

    const int numFFT = 1 + fftSize / 2;

    std::vector<std::thread> workers(numThreads);
    for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
        workers[threadIndex] = std::thread([&](int threadIdx) {
            std::vector<float> fftInput(fftSize);
            for (int i = 0; i < fftSize; i++) {
                fftInput[i] = 0.0;
            }

            std::vector<float> fftOutput(2 * fftSize);

            for (int i = threadIdx; i < melSpectrogram.numFrames; i += numThreads) {
                const int offset = i * fftStep;

                // Apply Hanning window
                for (int j = 0; j < fftSize; j++) {
                    if (offset + j < numSamples) {
                        fftInput[j] = hann[j] * samples[offset + j];
                    } else {
                        fftInput[j] = 0.0;
                    }
                }

                // FFT -> magnitude^2
                fft(fftInput, fftOutput);

                for (int j = 0; j < fftSize; j++) {
                    fftOutput[j] = (fftOutput[2 * j + 0] * fftOutput[2 * j + 0] +
                                    fftOutput[2 * j + 1] * fftOutput[2 * j + 1]);
                }
                for (int j = 1; j < fftSize / 2; j++) {
                    fftOutput[j] += fftOutput[fftSize - j];
                }

                // Mel spectrogram
                for (int j = 0; j < numMelFilters; j++) {
                    double sum = 0.0;

                    for (int k = 0; k < numFFT; k++) {
                        sum += fftOutput[k] * filters.data[j * numFFT + k];
                    }
                    if (sum < 1e-10) {
                        sum = 1e-10;
                    }

                    sum = log10(sum);

                    melSpectrogram.data[j * melSpectrogram.numFrames + i] = sum;
                }
            }
        },threadIndex);
    }

    for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
        workers[threadIndex].join();
    }

    // Clamping and normalization
    double maxVal = -1e20;
    for (int i = 0; i < melSpectrogram.numMelFilters * melSpectrogram.numFrames; i++) {
        if (melSpectrogram.data[i] > maxVal) {
            maxVal = melSpectrogram.data[i];
        }
    }
    maxVal -= 8.0;

    for (int i = 0; i < melSpectrogram.numMelFilters * melSpectrogram.numFrames; i++) {
        if (melSpectrogram.data[i] < maxVal) {
            melSpectrogram.data[i] = maxVal;
        }

        melSpectrogram.data[i] = (melSpectrogram.data[i] + 4.0) / 4.0;
    }

    return true;
}
