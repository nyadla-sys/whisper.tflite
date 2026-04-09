package com.whisper.android.tflitecpp.asr;

public interface IRecorderListener {
    void onUpdateReceived(String message);

    void onDataReceived(float[] samples);
}
