package com.whisper.android.tflitecpp;

import android.Manifest;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.whisper.android.tflitecpp.asr.IRecorderListener;
import com.whisper.android.tflitecpp.asr.IWhisperListener;
import com.whisper.android.tflitecpp.asr.Recorder;
import com.whisper.android.tflitecpp.asr.Whisper;
import com.whisper.android.tflitecpp.utils.ModelDownloader;
import com.whisper.android.tflitecpp.utils.WaveUtil;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private TextView tvStatus;
    private TextView tvResult;
    private ImageView fabMic;
    private FloatingActionButton fabCopy;
    private ExtendedFloatingActionButton fabModel;

    private static final String HF_BASE = "https://huggingface.co/nyadla-sys/whisper-tiny.en.tflite/resolve/main/";

    private static final String[][] MODELS = {
        // {label, filename, vocabFile, multilingual("1"/"0"), downloadUrl or null}
        {"tiny.en",  "whisper-tiny-en.tflite",  "filters_vocab_en.bin",           "0", null},
        {"small.en", "whisper-small.en.tflite",  "filters_vocab_en.bin",           "0", HF_BASE + "whisper-small.en.tflite"},
        {"small",    "whisper-small.tflite",     "filters_vocab_multilingual.bin", "1", HF_BASE + "whisper-small.tflite"},
    };
    private int currentModelIndex = 0;

    private Whisper mWhisper = null;
    private Recorder mRecorder = null;
    private MediaPlayer mPlayer = null;
    private LinearLayout audioBar;
    private ImageButton btnPlay;
    private SeekBar seekBar;
    private TextView tvDuration;
    private String lastRecordingPath = null;

    private final Handler handler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvStatus = findViewById(R.id.tvStatus);
        tvResult = findViewById(R.id.tvResult);
        fabMic   = findViewById(R.id.fabMic);
        fabCopy  = findViewById(R.id.fabCopy);
        fabModel = findViewById(R.id.fabModel);
        fabModel.setText("tiny.en");

        audioBar   = findViewById(R.id.audioBar);
        btnPlay    = findViewById(R.id.btnPlay);
        seekBar    = findViewById(R.id.seekBar);
        tvDuration = findViewById(R.id.tvDuration);

        btnPlay.setOnClickListener(v -> togglePlayback());

        // Model selection popup
        fabModel.setOnClickListener(v -> {
            if (mWhisper != null && mWhisper.isInProgress()) return;
            if (mRecorder != null && mRecorder.isInProgress()) return;
            PopupMenu popup = new PopupMenu(this, v);
            for (int i = 0; i < MODELS.length; i++) {
                popup.getMenu().add(0, i, i, MODELS[i][0]);
            }
            popup.setOnMenuItemClickListener(item -> {
                currentModelIndex = item.getItemId();
                fabModel.setText(MODELS[currentModelIndex][0]);
                initWhisper();
                return true;
            });
            popup.show();
        });

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
                    handler.post(() -> fabMic.setImageResource(R.drawable.ic_stop_foreground));
                } else if (message.equals(Recorder.MSG_RECORDING_DONE)) {
                    handler.post(() -> {
                        fabMic.setImageResource(R.drawable.ic_mic_foreground);
                        lastRecordingPath = getFilePath(WaveUtil.RECORDING_FILE);
                        showAudioBar();
                        startTranscription(lastRecordingPath);
                    });
                }
            }

            @Override
            public void onDataReceived(float[] samples) {}
        });

        checkRecordPermission();
    }

    private void initWhisper() {
        String[] m = MODELS[currentModelIndex];
        String modelPath = getFilePath(m[1]);
        String vocabPath = getFilePath(m[2]);
        boolean isMultilingual = m[3].equals("1");

        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            downloadModel(currentModelIndex);
            return;
        }

        tvStatus.setText("Loading " + m[0] + " model...");

        if (mWhisper == null) {
            mWhisper = new Whisper(this);
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
                String cleaned = result.replaceAll("<\\|[^|]*\\|>", "")
                                       .replaceAll("\\[_[A-Z]+_\\]", "")
                                       .replaceAll("\\[_TT_\\d+\\]", "")
                                       .replaceAll("\\[_extra_token_\\d+\\]", "")
                                       .trim();
                handler.post(() -> tvResult.append(cleaned));
            }
        });
        }
        mWhisper.loadModel(modelPath, vocabPath, isMultilingual);
    }

    private void downloadModel(int modelIndex) {
        String[] m = MODELS[modelIndex];
        String url = m[4];
        if (url == null) {
            tvStatus.setText("Error: " + m[0] + " model not found");
            return;
        }
        fabModel.setEnabled(false);
        fabMic.setEnabled(false);
        tvStatus.setText("Downloading " + m[0] + " model (0%)...");

        ModelDownloader.download(url, m[1], getFilesDir(), new ModelDownloader.Callback() {
            @Override
            public void onProgress(int percent) {
                tvStatus.setText("Downloading " + m[0] + " model (" + percent + "%)...");
            }

            @Override
            public void onComplete(File file) {
                fabModel.setEnabled(true);
                fabMic.setEnabled(true);
                initWhisper();
            }

            @Override
            public void onError(String error) {
                fabModel.setEnabled(true);
                fabMic.setEnabled(true);
                tvStatus.setText("Download failed: " + error);
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

    private void showAudioBar() {
        audioBar.setVisibility(View.VISIBLE);
        try {
            if (mPlayer != null) { mPlayer.release(); mPlayer = null; }
            mPlayer = new MediaPlayer();
            mPlayer.setDataSource(lastRecordingPath);
            mPlayer.prepare();
            int dur = mPlayer.getDuration();
            seekBar.setMax(dur);
            seekBar.setProgress(0);
            tvDuration.setText("0:00 / " + formatTime(dur));
            btnPlay.setImageResource(android.R.drawable.ic_media_play);

            mPlayer.setOnCompletionListener(mp -> {
                btnPlay.setImageResource(android.R.drawable.ic_media_play);
                seekBar.setProgress(0);
                tvDuration.setText("0:00 / " + formatTime(dur));
            });

            seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                public void onProgressChanged(SeekBar sb, int progress, boolean fromUser) {
                    if (fromUser && mPlayer != null) mPlayer.seekTo(progress);
                }
                public void onStartTrackingTouch(SeekBar sb) {}
                public void onStopTrackingTouch(SeekBar sb) {}
            });
        } catch (IOException e) {
            Log.e(TAG, "MediaPlayer error", e);
        }
    }

    private void togglePlayback() {
        if (mPlayer == null) return;
        if (mPlayer.isPlaying()) {
            mPlayer.pause();
            btnPlay.setImageResource(android.R.drawable.ic_media_play);
        } else {
            mPlayer.start();
            btnPlay.setImageResource(android.R.drawable.ic_media_pause);
            updateSeekBar();
        }
    }

    private void updateSeekBar() {
        if (mPlayer != null && mPlayer.isPlaying()) {
            int pos = mPlayer.getCurrentPosition();
            seekBar.setProgress(pos);
            tvDuration.setText(formatTime(pos) + " / " + formatTime(mPlayer.getDuration()));
            handler.postDelayed(this::updateSeekBar, 200);
        }
    }

    private String formatTime(int ms) {
        int s = ms / 1000;
        return s / 60 + ":" + String.format("%02d", s % 60);
    }

    private String getFilePath(String name) {
        return new File(getFilesDir(), name).getAbsolutePath();
    }
}
