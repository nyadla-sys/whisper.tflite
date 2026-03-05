package com.whispertflite.utils;

// Feature: whisper-android-audio-inference, Property 10: WAV header round-trip

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property 10: WAV header round-trip
 *
 * For any byte array of 16-bit PCM samples written by WaveUtil.createWaveFile()
 * with sampleRate=16000, channels=1, bytesPerSample=2, reading the file back
 * with WaveUtil.getSamples() must return a float[] of equivalent length and
 * values (within floating-point normalization tolerance).
 *
 * Validates: Requirements 7.1, 7.4
 */
public class WaveUtilPropertyTest {

    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNELS = 1;
    private static final int BYTES_PER_SAMPLE = 2;
    // Normalization tolerance: 1/32768 ≈ 3.05e-5
    private static final double TOLERANCE = 1.0 / 32768.0;

    /**
     * Property 10: WAV header round-trip
     * Validates: Requirements 7.1, 7.4
     */
    @Property(tries = 100)
    void wavHeaderRoundTrip(@ForAll @Size(min = 2, max = 1000) byte[] rawSamples) throws IOException {
        // Ensure even length (16-bit samples = 2 bytes each)
        int len = rawSamples.length & ~1; // round down to even
        if (len == 0) return;

        byte[] samples = new byte[len];
        System.arraycopy(rawSamples, 0, samples, 0, len);

        File tmpFile = File.createTempFile("wavtest", ".wav");
        tmpFile.deleteOnExit();
        String path = tmpFile.getAbsolutePath();

        try {
            // Write WAV file
            WaveUtil.createWaveFile(path, samples, SAMPLE_RATE, CHANNELS, BYTES_PER_SAMPLE);

            // Read back
            float[] result = WaveUtil.getSamples(path);

            int expectedSamples = len / BYTES_PER_SAMPLE;
            assertEquals(expectedSamples, result.length,
                    "getSamples() must return same number of samples as written");

            // Verify each sample value matches within normalization tolerance
            java.nio.ByteBuffer buf = java.nio.ByteBuffer.wrap(samples);
            buf.order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < expectedSamples; i++) {
                short s = buf.getShort();
                double expected = s / 32768.0;
                assertEquals(expected, result[i], TOLERANCE,
                        "Sample " + i + " value mismatch after round-trip");
            }
        } finally {
            Files.deleteIfExists(tmpFile.toPath());
        }
    }

    /**
     * Non-RIFF file returns empty array.
     * Validates: Requirements 7.4
     */
    @Property(tries = 100)
    void nonRiffFileReturnsEmpty(@ForAll @Size(min = 44, max = 200) byte[] garbage) throws IOException {
        // Ensure first 4 bytes are NOT "RIFF"
        garbage[0] = 'X'; garbage[1] = 'X'; garbage[2] = 'X'; garbage[3] = 'X';

        File tmpFile = File.createTempFile("nonriff", ".wav");
        tmpFile.deleteOnExit();
        try {
            Files.write(tmpFile.toPath(), garbage);
            float[] result = WaveUtil.getSamples(tmpFile.getAbsolutePath());
            assertEquals(0, result.length, "Non-RIFF file must return empty float[]");
        } finally {
            Files.deleteIfExists(tmpFile.toPath());
        }
    }

    /**
     * RECORDING_FILE constant must equal "MicInput.wav".
     * Validates: Requirements 2.4
     */
    @Example
    void recordingFileConstant() {
        assertEquals("MicInput.wav", WaveUtil.RECORDING_FILE);
    }
}
