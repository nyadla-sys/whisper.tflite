package com.whispertflite;

import android.Manifest;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.whispertflite.asr.IRecorderListener;
import com.whispertflite.asr.IWhisperListener;
import com.whispertflite.asr.Recorder;
import com.whispertflite.asr.Whisper;
import com.whispertflite.utils.WaveUtil;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private TextView tvStatus;
    private TextView tvResult;
    private FloatingActionButton fabMic;
    private FloatingActionButton fabCopy;

    private Whisper mWhisper = null;
    private Recorder mRecorder = null;

    private final Handler handler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvStatus = findViewById(R.id.tvStatus);
        tvResult = findViewById(R.id.tvResult);
        fabMic   = findViewById(R.id.fabMic);
        fabCopy  = findViewById(R.id.fabCopy);

        // Copy FAB
        fabCopy.setOnClickListener(v -> {
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            clipboard.setPrimaryClip(ClipData.newPlainText("Transcription", tvResult.getText()));
        });

        // Mic FAB — tap to record, tap again to stop and transcribe
        fabMic.setOnClickListener(v -> {
            if (mRecorder != null && mRecorder.isInProgress()) {
                stopRecordingAndTranscribe();
            } else {
                startRecording();
            }
        });

        // Copy assets and init model
        copyAssetsWithExtensionsToDataFolder(this, new String[]{"pcm", "bin", "wav", "tflite"});
        initWhisper();

        mRecorder = new Recorder(this);
        mRecorder.setListener(new IRecorderListener() {
            @Override
            public void onUpdateReceived(String message) {
                handler.post(() -> tvStatus.setText(message));
                if (message.equals(Recorder.MSG_RECORDING)) {
                    // Switch to stop icon while recording
                    handler.post(() -> fabMic.setImageResource(R.drawable.ic_stop));
                } else if (message.equals(Recorder.MSG_RECORDING_DONE)) {
                    // Restore mic icon, then auto-transcribe
                    handler.post(() -> {
                        fabMic.setImageResource(R.drawable.ic_mic);
                        String waveFilePath = getFilePath(WaveUtil.RECORDING_FILE);
                        startTranscription(waveFilePath);
                    });
                }
            }

            @Override
            public void onDataReceived(float[] samples) {}
        });

        checkRecordPermission();
    }

    private void initWhisper() {
        String modelPath = getFilePath("whisper-tiny-en.tflite");
        String vocabPath = getFilePath("filters_vocab_en.bin");

        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            tvStatus.setText("Error: model not found");
            return;
        }

        mWhisper = new Whisper(this);
        mWhisper.loadModel(modelPath, vocabPath, false);
        mWhisper.setListener(new IWhisperListener() {
            @Override
            public void onUpdateReceived(String message) {
                handler.post(() -> tvStatus.setText(message));
                if (message.equals(Whisper.MSG_PROCESSING)) {
                    handler.post(() -> tvResult.setText(""));
                }
            }

            @Override
            public void onResultReceived(String result) {
                handler.post(() -> tvResult.append(result));
            }
        });
    }

    private void startRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            tvStatus.setText("Microphone permission required");
            return;
        }
        mRecorder.setFilePath(getFilePath(WaveUtil.RECORDING_FILE));
        mRecorder.start();
    }

    private void stopRecordingAndTranscribe() {
        mRecorder.stop();
        // transcription is triggered automatically in onUpdateReceived(MSG_RECORDING_DONE)
    }

    private void startTranscription(String waveFilePath) {
        if (mWhisper == null || mWhisper.isInProgress()) return;
        mWhisper.setFilePath(waveFilePath);
        mWhisper.setAction(Whisper.ACTION_TRANSCRIBE);
        mWhisper.start();
    }

    private void checkRecordPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 0);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults.length == 0 || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            handler.post(() -> tvStatus.setText("Microphone permission denied"));
        }
    }

    private static void copyAssetsWithExtensionsToDataFolder(Context context, String[] extensions) {
        AssetManager assetManager = context.getAssets();
        try {
            String destFolder = context.getFilesDir().getAbsolutePath();
            for (String ext : extensions) {
                for (String name : assetManager.list("")) {
                    if (!name.endsWith("." + ext)) continue;
                    File out = new File(destFolder, name);
                    if (out.exists()) continue;
                    try (InputStream in = assetManager.open(name);
                         OutputStream os = new FileOutputStream(out)) {
                        byte[] buf = new byte[4096];
                        int n;
                        while ((n = in.read(buf)) != -1) os.write(buf, 0, n);
                    }
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Asset copy error", e);
        }
    }

    private String getFilePath(String name) {
        return new File(getFilesDir(), name).getAbsolutePath();
    }
}
