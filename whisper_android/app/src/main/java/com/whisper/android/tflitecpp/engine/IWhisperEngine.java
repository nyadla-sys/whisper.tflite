package com.whisper.android.tflitecpp.engine;

import com.whisper.android.tflitecpp.asr.IWhisperListener;

import java.io.IOException;

public interface IWhisperEngine {
    boolean isInitialized();
    void interrupt();
    void setUpdateListener(IWhisperListener listener);
    boolean initialize(String modelPath, String vocabPath, boolean multilingual) throws IOException;
    String transcribeFile(String wavePath);
    String transcribeBuffer(float[] samples);

    //String getTranslation(String wavePath);
}
