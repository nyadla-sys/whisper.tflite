package com.example.tfliteaudio;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends AppCompatActivity {

    private final String TAG = "MainActivity";

    private Button btnMicRec;
    private Button btnTranscb;
    private TextView tvResult;
    private Handler mHandler;
    private String mSelectedFile;
    private AudioRecord mAudioRecord;
    private Thread mRecordingThread;
    private Thread mTranscriptionThread;

    private final String mRecordedFile = "MicInput.wav";
    private final int mBufferSize = WaveUtil.BUFFER_SIZE_30_SEC;
    private final TFLiteEngine mTFLiteEngine = new TFLiteEngine();
    private final AtomicBoolean mRecordingInProgress = new AtomicBoolean(false);
    private final AtomicBoolean mTranscriptionInProgress = new AtomicBoolean(false);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main_one);
        mHandler = new Handler(Looper.getMainLooper());
        tvResult = findViewById(R.id.tvResult);

        // Implementation of transcribe button functionality
        btnTranscb = findViewById(R.id.btnTranscb);
        btnTranscb.setOnClickListener(v -> {
            if (mRecordingInProgress.get())
                stopRecording();

            if (!mTranscriptionInProgress.get())
                startTranscription();
            else
                Log.d(TAG, "Transcription is already in progress...!");
        });

        // Implementation of record button functionality
        btnMicRec = findViewById(R.id.btnMicRecord);
        btnMicRec.setOnClickListener(v -> {
            if (!mRecordingInProgress.get())
                startRecording();
            else
                stopRecording();
        });

        // Implementation of file spinner functionality
        ArrayList<String> files = new ArrayList<>();
//        files.add(mRecordedFile);
        try {
            String[] assetFiles = getAssets().list("");
            for (String file : assetFiles) {
                if (file.endsWith(".wav"))
                    files.add(file);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        String[] fileArray = new String[files.size()];
        files.toArray(fileArray);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, fileArray);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        Spinner spinner = findViewById(R.id.spnrFiles);
        spinner.setAdapter(adapter);
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                mSelectedFile = fileArray[position];
                if (mSelectedFile.equals(mRecordedFile))
                    btnMicRec.setVisibility(View.VISIBLE);
                else
                    btnMicRec.setVisibility(View.GONE);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        // Call the method to copy specific file types from assets to data folder
        String[] extensionsToCopy = {"pcm", "bin", "wav", "tflite"};
        copyAssetsWithExtensionsToDataFolder(this, extensionsToCopy);

        // Assume this Activity is the current activity, check record permission
        checkRecordPermission();
    }

    private void checkRecordPermission() {
        int permission = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO);
        if (permission == PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "Record permission is granted");
        } else {
            Log.d(TAG, "Requesting record permission");
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 0);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
            Log.d(TAG, "Record permission is granted");
        else
            Log.d(TAG, "Record permission is not granted");
    }

    private void startRecording() {
        checkRecordPermission();

        mAudioRecord = new AudioRecord(
                MediaRecorder.AudioSource.DEFAULT,
                WaveUtil.SAMPLE_RATE,
                WaveUtil.CHANNEL_CONFIG,
                WaveUtil.AUDIO_FORMAT,
                mBufferSize
        );
        mAudioRecord.startRecording();
        mRecordingInProgress.set(true);
        mRecordingThread = new Thread(new RecordingRunnable(), "Recording Thread");
        mRecordingThread.start();
        mHandler.post(() -> btnMicRec.setText(getString(R.string.stop)));
    }

    private void stopRecording() {
        if (mAudioRecord == null)
            return;

        mRecordingInProgress.set(false);
        mAudioRecord.stop();
        mAudioRecord.release();
        mAudioRecord = null;
        try {
            mRecordingThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        mRecordingThread = null;
        mHandler.post(() -> btnMicRec.setText(getString(R.string.record)));
    }

    private void startTranscription() {
        mTranscriptionInProgress.set(true);
        mTranscriptionThread = new Thread(new TranscriptionRunnable(), "Transcription Thread");
        mTranscriptionThread.start();
    }

    // Copies specified asset to app's files directory and returns its absolute path.
    private String getFilePath(String assetName) {
        File outfile = new File(getFilesDir(), assetName);
        if (!outfile.exists()) {
            Log.d(TAG, "File not found - " + outfile.getAbsolutePath());
        }

        Log.d(TAG, "Returned asset path: " + outfile.getAbsolutePath());
        return outfile.getAbsolutePath();
    }

    public static void copyAssetsWithExtensionsToDataFolder(Context context, String[] extensions) {
            AssetManager assetManager = context.getAssets();

            try {
                // Specify the destination directory in the app's data folder
                String destFolder = context.getFilesDir().getAbsolutePath();

                for (String extension : extensions) {
                    // List all files in the assets folder with the specified extension
                    String[] assetFiles = assetManager.list("");
                    for (String assetFileName : assetFiles) {
                        if (assetFileName.endsWith("." + extension)) {
                            InputStream in = assetManager.open(assetFileName);
                            File outFile = new File(destFolder, assetFileName);
                            OutputStream out = new FileOutputStream(outFile);

                            // Copy the file from assets to the data folder
                            byte[] buffer = new byte[1024];
                            int read;
                            while ((read = in.read(buffer)) != -1) {
                                out.write(buffer, 0, read);
                            }

                            in.close();
                            out.flush();
                            out.close();
                        }
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    private class RecordingRunnable implements Runnable {
        @Override
        public void run() {
            try {
                ByteBuffer buffer = ByteBuffer.allocateDirect(mBufferSize);
                if (mRecordingInProgress.get()) {
                    final String recordMsg = "Recording audio for 30 sec...";
                    mHandler.post(() -> tvResult.setText(recordMsg));
                    Log.d(TAG, recordMsg);

                    int result = mAudioRecord.read(buffer, mBufferSize);
                    if (result < 0) {
                        Log.d(TAG, "AudioRecord read error!!!");
                    }
                }

                // call stopRecording on thread to avoid deadlock in this function
                new Thread(() -> {
                    if (mRecordingInProgress.get())
                        stopRecording();
                }).start();

                // Write samples to wav file
                float[] samples = WaveUtil.getSamples(buffer.array(), WaveUtil.BYTES_PER_SAMPLE);
                String wavePath = getFilesDir().toString() + File.separator + mRecordedFile;
                WaveUtil.createWaveFile(wavePath, samples);
                Log.d(TAG, "Recorded file: " + wavePath);
                mHandler.post(() -> tvResult.setText(getString(R.string.recording_is_completed)));
            } catch (Exception e) {
                throw new RuntimeException("Writing of recorded audio failed", e);
            }
        }
    }

    private class TranscriptionRunnable implements Runnable {
        @Override
        public void run() {
            try {
                // Initialize TFLiteEngine
                if (!mTFLiteEngine.isInitialized()) {
                    // Update progress to UI thread
                    mHandler.post(() -> tvResult.setText(getString(R.string.loading_model_and_vocab)));

                    // set true for multilingual support
                    // whisper.tflite => not multilingual
                    // whisper-small.tflite => multilingual
                    // whisper-tiny.tflite => multilingual
                    boolean isMultilingual = true;

                    // Get Model and vocab file paths
                    String modelPath;
                    String vocabPath;
                    if (isMultilingual) {
                        modelPath = getFilePath("whisper-tiny.tflite");
                        vocabPath = getFilePath("filters_vocab_multilingual.bin");
                    } else {
                        modelPath = getFilePath("whisper-tiny-en.tflite");
                        vocabPath = getFilePath("filters_vocab_gen.bin");
                    }

                    mTFLiteEngine.initialize(isMultilingual, vocabPath, modelPath);
                }

                // Get Transcription
                if (mTFLiteEngine.isInitialized()) {
                    String wavePath = getFilePath(mSelectedFile);
                    Log.d(TAG, "WaveFile: " + wavePath);

                    if (new File(wavePath).exists()) {
                        // Update progress to UI thread
                        mHandler.post(() -> tvResult.setText(getString(R.string.transcribing)));
                        long startTime = System.currentTimeMillis();

                        // Get transcription from wav file
                        String result = mTFLiteEngine.getTranscription(wavePath);

                        // Display output result
                        mHandler.post(() -> tvResult.setText(result));
                        long endTime = System.currentTimeMillis();
                        long timeTaken = endTime - startTime;
                        Log.d(TAG, "Time Taken for transcription: " + timeTaken + "ms");
                        Log.d(TAG, "Result len: " + result.length() + ", Result: " + result);
                    } else {
                        mHandler.post(() -> tvResult.setText(getString(R.string.input_file_doesn_t_exist)));
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
                Log.e(TAG, "Error..", e);
                mHandler.post(() -> tvResult.setText(e.getMessage()));
            } finally {
                mTranscriptionInProgress.set(false);
            }
        }
    }
}
