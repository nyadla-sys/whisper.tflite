package com.whisper.android.tflitecpp.asr;

public interface IWhisperListener {
    void onUpdateReceived(String message);
    void onResultReceived(String result);
}
