// Feature: whisper-android-audio-inference
// Property 4: transcribeFile returns a non-null string for valid WAV input
// Property 11: Non-16kHz WAV produces empty transcription

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// Include wav_util directly (header-only)
#include "wav_util.h"

// ---------------------------------------------------------------------------
// WAV writing helper
// ---------------------------------------------------------------------------

/// Write a WAV file compatible with readWAVFile() in wav_util.h.
/// wav_util.h reads a 36-byte WAVHeader (no data chunk header), then reads
/// wav_size bytes of PCM data. So we write the header with wav_size = data_size
/// and follow it immediately with PCM samples (no "data" sub-chunk header).
static bool writeWAV(const std::string& path,
                     const std::vector<int16_t>& samples,
                     uint32_t sampleRate)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    // WAVHeader as defined in wav_util.h (36 bytes, no data sub-chunk header)
    struct WavHdr {
        char     riff_header[4];   // "RIFF"
        uint32_t wav_size;         // set to data size (wav_util quirk)
        char     wave_header[4];   // "WAVE"
        char     fmt_header[4];    // "fmt "
        uint32_t fmt_chunk_size;   // 16
        uint16_t audio_format;     // 1 = PCM
        uint16_t num_channels;     // 1 = mono
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample;
    } __attribute__((packed));

    WavHdr hdr;
    std::memcpy(hdr.riff_header, "RIFF", 4);
    std::memcpy(hdr.wave_header, "WAVE", 4);
    std::memcpy(hdr.fmt_header,  "fmt ", 4);

    hdr.fmt_chunk_size  = 16;
    hdr.audio_format    = 1;
    hdr.num_channels    = 1;
    hdr.sample_rate     = sampleRate;
    hdr.bits_per_sample = 16;
    hdr.block_align     = static_cast<uint16_t>(hdr.num_channels * (hdr.bits_per_sample / 8));
    hdr.byte_rate       = sampleRate * hdr.block_align;
    // wav_util.h uses wav_size as the data byte count
    hdr.wav_size        = static_cast<uint32_t>(samples.size() * sizeof(int16_t));

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(samples.data()),
            static_cast<std::streamsize>(samples.size() * sizeof(int16_t)));
    return f.good();
}

// ---------------------------------------------------------------------------
// Property 4: readWAVFile returns a non-empty vector for a valid 16 kHz WAV
// Validates: Requirements 4.2, 7.2
// ---------------------------------------------------------------------------

// Feature: whisper-android-audio-inference, Property 4: transcribeFile returns a non-null string for valid WAV input
class Property4_ValidWAVReturnsNonNull : public ::testing::TestWithParam<int> {};

TEST_P(Property4_ValidWAVReturnsNonNull, ReadWAVFileReturnsNonEmptyForValid16kHzWAV)
{
    int seed = GetParam();
    std::mt19937 rng(static_cast<unsigned>(seed));
    // Generate between 160 and 16000 random PCM samples (10 ms – 1 s)
    std::uniform_int_distribution<int> lenDist(160, 16000);
    std::uniform_int_distribution<int16_t> sampleDist(
        std::numeric_limits<int16_t>::min(),
        std::numeric_limits<int16_t>::max());

    int numSamples = lenDist(rng);
    std::vector<int16_t> pcm(numSamples);
    for (auto& s : pcm) s = sampleDist(rng);

    // Write a valid 16 kHz mono WAV to a temp file
    std::string tmpPath = "/tmp/prop4_test_" + std::to_string(seed) + ".wav";
    ASSERT_TRUE(writeWAV(tmpPath, pcm, 16000))
        << "Failed to write test WAV file: " << tmpPath;

    // readWAVFile must return a non-empty vector (no crash, no null)
    std::vector<float> result = readWAVFile(tmpPath.c_str());
    EXPECT_FALSE(result.empty())
        << "readWAVFile returned empty vector for a valid 16 kHz WAV (seed=" << seed << ")";

    // Cleanup
    std::remove(tmpPath.c_str());
}

// Run 100 iterations with different seeds
INSTANTIATE_TEST_SUITE_P(
    Property4Iterations,
    Property4_ValidWAVReturnsNonNull,
    ::testing::Range(0, 100));

// ---------------------------------------------------------------------------
// Property 11: readWAVFile returns empty vector for non-16kHz WAV
// Validates: Requirements 7.3
// ---------------------------------------------------------------------------

// Feature: whisper-android-audio-inference, Property 11: Non-16kHz WAV produces empty transcription
class Property11_NonStandardRateReturnsEmpty : public ::testing::TestWithParam<uint32_t> {};

TEST_P(Property11_NonStandardRateReturnsEmpty, ReadWAVFileReturnsEmptyForNon16kHzWAV)
{
    uint32_t sampleRate = GetParam();
    ASSERT_NE(sampleRate, 16000u) << "Test setup error: sample rate must not be 16000";

    // Write a short WAV with the non-standard sample rate
    std::vector<int16_t> pcm(sampleRate / 10, 0);  // 100 ms of silence
    std::string tmpPath = "/tmp/prop11_test_" + std::to_string(sampleRate) + ".wav";
    ASSERT_TRUE(writeWAV(tmpPath, pcm, sampleRate))
        << "Failed to write test WAV file: " << tmpPath;

    // readWAVFile must return an empty vector for non-16kHz input
    std::vector<float> result = readWAVFile(tmpPath.c_str());
    EXPECT_TRUE(result.empty())
        << "readWAVFile should return empty vector for sample rate "
        << sampleRate << " Hz (not 16000 Hz)";

    // Cleanup
    std::remove(tmpPath.c_str());
}

// Sample rates in [8000, 48000] \ {16000} — 100 representative values
static std::vector<uint32_t> nonStandardRates()
{
    std::vector<uint32_t> rates;
    // Common non-16kHz rates
    for (uint32_t r : {8000u, 11025u, 12000u, 22050u, 24000u, 32000u,
                       44100u, 48000u}) {
        rates.push_back(r);
    }
    // Fill up to 100 with random rates in [8000, 48000] \ {16000}
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(8000, 48000);
    while (rates.size() < 100) {
        uint32_t r = dist(rng);
        if (r != 16000u) rates.push_back(r);
    }
    return rates;
}

INSTANTIATE_TEST_SUITE_P(
    Property11Iterations,
    Property11_NonStandardRateReturnsEmpty,
    ::testing::ValuesIn(nonStandardRates()));
