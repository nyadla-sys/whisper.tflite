package com.whisper.android.tflitecpp;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.widget.Chronometer;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.google.android.material.button.MaterialButton;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private ImageView recordAudioButton;
    private TextView transcribeResultTxt;
    private WavAudioRecorder mRecorder = null;
    private static String fileName = null;
    private Chronometer record_chronometer;
    private final static String TAG = "TFLiteASRDemo";
    private MediaRecorder recorder = null;
    private static File rootDir = null;
    private static int sRate;   // Records the best sample rate supported by the device
    private static MaterialButton transcribeButton = null;
    // Requesting permission to RECORD_AUDIO
    private boolean permissionToRecordAccepted = false;
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private String[] permissions = {Manifest.permission.RECORD_AUDIO};
    int spittedAudioFileCount = 0;


    // Requesting permission to RECORD_AUDIO
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            permissionToRecordAccepted = grantResults[0] == PackageManager.PERMISSION_GRANTED;
        }
        if (!permissionToRecordAccepted) finish();
    }

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getSupportActionBar().hide();
        initView();
        rootDir = new File(Environment.getExternalStorageDirectory(), "Documents/Whisper/" + getDate());
        if (!rootDir.exists()) {
            rootDir.mkdirs(); // Create necessary directories
        }
        fileName = rootDir.getAbsolutePath() + "/android_record.wav";
        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION);

        recordAudioButton.setImageResource(R.drawable.ic_mic_foreground);

        recordAudioButton.setOnClickListener(view -> {
            if (mRecorder == null) {
                mRecorder = WavAudioRecorder.getInstanse(MainActivity.this);
            }
            if (WavAudioRecorder.State.INITIALIZING == mRecorder.getState()) {
//                Log.d(TAG, "INITIALIZING" + fileName);
                // Set the chronometer to count up from zero and start it running and counting up
                record_chronometer.setBase(SystemClock.elapsedRealtime());
                record_chronometer.setCountDown(false);
                record_chronometer.start();
                mRecorder.setOutputFile(fileName);
                mRecorder.prepare();
                mRecorder.start();
                recordAudioButton.setImageResource(R.drawable.ic_stop_foreground);
            } else if (WavAudioRecorder.State.ERROR == mRecorder.getState()) {
                Log.e(TAG, "ERROR");
                mRecorder.release();
                mRecorder = WavAudioRecorder.getInstanse(MainActivity.this);
                mRecorder.setOutputFile(fileName);
                recordAudioButton.setImageResource(R.drawable.ic_mic_foreground);
            } else {
                Log.d(TAG, "On Record Stop Click");
                stopRecording();
                startTranscribe();
            }
            //TODO: for debugging purpose only
//            transcribeButton.setOnClickListener(v -> startTranscribe());
        });
    }

    private String getDate() {
        long tsLong = System.currentTimeMillis() / 1000;
        Date date = new Date(tsLong * 1000L);  // Convert seconds to milliseconds
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss", Locale.getDefault()); // Define the desired date format
        return sdf.format(date);  // Format the date and return it as a string
    }

    private void stopRecording() {
        record_chronometer.stop();
        mRecorder.stop();
        mRecorder.reset(MainActivity.this);
        mRecorder.release();
        mRecorder = WavAudioRecorder.getInstanse(MainActivity.this);
        recordAudioButton.setImageResource(R.drawable.ic_mic_foreground);
        splitAudioFile(fileName);
    }

    private void startTranscribe() {
        try {
            StringBuilder res = new StringBuilder();
            for (int i = 0; i < spittedAudioFileCount; i++) {
                String audioFileName = rootDir.getAbsolutePath() + "/audio_files/audio_" + (i + 1) + ".wav";
                String result = loadModelJNI(getAssets(), audioFileName, 1);
                res.append(result);
            }
            writeToFile(String.valueOf(res));
            transcribeResultTxt.setText(res);
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
        }
    }

    private void writeWavHeader(FileOutputStream outputStream, int audioDataLength) throws IOException {
        // Constants for WAV header
        int sampleRate = sRate;
        int bitsPerSample = 16;
        int numChannels = 1;  // Assuming mono audio

        // Calculate header fields
        int audioFormat = 1;  // PCM
        int byteRate = sampleRate * numChannels * bitsPerSample / 8;
        int blockAlign = numChannels * bitsPerSample / 8;

        // Write WAV header
        outputStream.write("RIFF".getBytes());
        outputStream.write(intToByteArray(36 + audioDataLength), 0, 4);
        outputStream.write("WAVE".getBytes());
        outputStream.write("fmt ".getBytes());
        outputStream.write(intToByteArray(16), 0, 4);
        outputStream.write(shortToByteArray((short) audioFormat), 0, 2);
        outputStream.write(shortToByteArray((short) numChannels), 0, 2);
        outputStream.write(intToByteArray(sampleRate), 0, 4);
        outputStream.write(intToByteArray(byteRate), 0, 4);
        outputStream.write(shortToByteArray((short) blockAlign), 0, 2);
        outputStream.write(shortToByteArray((short) bitsPerSample), 0, 2);
        outputStream.write("data".getBytes());
        outputStream.write(intToByteArray(audioDataLength), 0, 4);
    }

    private byte[] intToByteArray(int value) {
        return new byte[]{
                (byte) (value),
                (byte) (value >> 8),
                (byte) (value >> 16),
                (byte) (value >> 24)
        };
    }

    private byte[] shortToByteArray(short value) {
        return new byte[]{
                (byte) (value),
                (byte) (value >> 8)
        };
    }

    private void splitAudioFile(String fileName) {

        File audioFile = new File(fileName);
        List<byte[]> audioChunks = new ArrayList<>();

        try (FileInputStream inputStream = new FileInputStream(audioFile)) {
            byte[] buffer = new byte[sRate * 60];   // 28 seconds audio chunk
            int bytesRead;
            int chunkNumber = 1;
            File outputDirectory = new File(rootDir, "audio_files");
            Log.d("Output Directory", outputDirectory.getAbsolutePath());

            if (!outputDirectory.exists()) {
                outputDirectory.mkdirs();
            }

            while ((bytesRead = inputStream.read(buffer)) != -1) {
                byte[] chunk = new byte[bytesRead];
                System.arraycopy(buffer, 0, chunk, 0, bytesRead);
                audioChunks.add(chunk);

                // Save the chunk as a separate audio file
                File outputFile = new File(outputDirectory, "audio_" + chunkNumber + ".wav");
                try (FileOutputStream outputStream = new FileOutputStream(outputFile)) {
                    writeWavHeader(outputStream, chunk.length);
                    outputStream.write(chunk);
                }

                chunkNumber++;
            }
            spittedAudioFileCount = chunkNumber-1;
            // Now 'audioChunks' contains the audio file split into chunks
            // Chunks are saved in the 'audio_files' subdirectory
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void writeToFile(String res) {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state)) {
            // Create file and write result to it
            File file = new File(rootDir, "TranscribeResult.txt");
            try (FileOutputStream fos = new FileOutputStream(file)) {
                fos.write(res.getBytes());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void initView() {
        recordAudioButton = findViewById(R.id.record);
        transcribeResultTxt = findViewById(R.id.result);
        record_chronometer = findViewById(R.id.record_chronometer);
//        transcribeButton = findViewById(R.id.transcribeBtn);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */

    // Load model by TF Lite C++ API
    private native String loadModelJNI(AssetManager assetManager, String fileName, int is_recorded);

    private native int freeModelJNI();

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (null != recorder) {
            recorder.release();
        }
        freeModelJNI();
    }

    public static class WavAudioRecorder {
        private final static int[] sampleRates = {44100, 22050, 16000, 11025, 8000};

        public static WavAudioRecorder getInstanse(Context context) {
            WavAudioRecorder result;
            int i = 2;
            do {
                result = new WavAudioRecorder(context, MediaRecorder.AudioSource.MIC,
                        sampleRates[i],
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT);
            } while ((++i < sampleRates.length) & (result.getState() != State.INITIALIZING));
            return result;
        }

        /**
         * INITIALIZING : recorder is initializing;
         * READY : recorder has been initialized, recorder not yet started
         * RECORDING : recording
         * ERROR : reconstruction needed
         * STOPPED: reset needed
         */
        public enum State {INITIALIZING, READY, RECORDING, ERROR, STOPPED}

        // The interval in which the recorded samples are output to the file
        // Used only in uncompressed mode
        private static final int TIMER_INTERVAL = 120;

        // Recorder used for uncompressed recording
        private AudioRecord audioRecorder = null;

        // Output file path
        private String filePath = null;

        // Recorder state; see State
        private State state;

        // File writer (only in uncompressed mode)
        private RandomAccessFile randomAccessWriter;

        // Number of channels, sample rate, sample size(size in bits), buffer size, audio source, sample size(see AudioFormat)
        private short nChannels;
        private short mBitsPersample;
        private int mBufferSize;
        private int mAudioSource;
        private int aFormat;

        // Number of frames/samples written to file on each output(only in uncompressed mode)
        private int mPeriodInFrames;

        // Buffer for output(only in uncompressed mode)
        private byte[] buffer;

        // Number of bytes written to file after header(only in uncompressed mode)
        // after stop() is called, this size is written to the header/data chunk in the wave file
        private int payloadSize;

        /**
         * Returns the state of the recorder in a WavAudioRecorder.State typed object.
         * Useful, as no exceptions are thrown.
         *
         * @return recorder state
         */
        public State getState() {
            return state;
        }

        private AudioRecord.OnRecordPositionUpdateListener updateListener = new AudioRecord.OnRecordPositionUpdateListener() {
            //	periodic updates on the progress of the record head
            public void onPeriodicNotification(AudioRecord recorder) {
                if (State.STOPPED == state) {
                    Log.d(WavAudioRecorder.this.getClass().getName(), "recorder stopped");
                    return;
                }
                int numOfBytes = audioRecorder.read(buffer, 0, buffer.length); // read audio data to buffer
                //			Log.d(WavAudioRecorder.this.getClass().getName(), state + ":" + numOfBytes);
                try {
                    randomAccessWriter.write(buffer);          // write audio data to file
                    payloadSize += buffer.length;
                } catch (IOException e) {
                    Log.e(WavAudioRecorder.class.getName(), "Error occurred in updateListener, recording is aborted");
                    e.printStackTrace();
                }
            }

            // reached a notification marker set by setNotificationMarkerPosition(int)
            public void onMarkerReached(AudioRecord recorder) {

            }
        };

        /**
         * Default constructor
         * <p>
         * Instantiates a new recorder
         * In case of errors, no exception is thrown, but the state is set to ERROR
         */
        public WavAudioRecorder(Context context, int audioSource, int sampleRate, int channelConfig, int audioFormat) {
            try {
                if (audioFormat == AudioFormat.ENCODING_PCM_16BIT) {
                    mBitsPersample = 16;
                } else {
                    mBitsPersample = 8;
                }

                if (channelConfig == AudioFormat.CHANNEL_IN_MONO) {
                    nChannels = 1;
                } else {
                    nChannels = 2;
                }

                mAudioSource = audioSource;
                sRate = sampleRate;
                aFormat = audioFormat;

                mPeriodInFrames = sampleRate * TIMER_INTERVAL / 1000;        //?
                mBufferSize = mPeriodInFrames * 2 * nChannels * mBitsPersample / 8;        //?
                if (mBufferSize < AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)) {
                    // Check to make sure buffer size is not smaller than the smallest allowed one
                    mBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat);
                    // Set frame period and timer interval accordingly
                    mPeriodInFrames = mBufferSize / (2 * mBitsPersample * nChannels / 8);
                    Log.w(WavAudioRecorder.class.getName(), "Increasing buffer size to " + mBufferSize);
                }

                if (ActivityCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                    // TODO: Consider calling
                    //    ActivityCompat#requestPermissions
                    // here to request the missing permissions, and then overriding
                    //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                    //                                          int[] grantResults)
                    // to handle the case where the user grants the permission. See the documentation
                    // for ActivityCompat#requestPermissions for more details.
                    return; //TODO;
                }
                audioRecorder = new AudioRecord(audioSource, sampleRate, channelConfig, audioFormat, mBufferSize);

                if (audioRecorder.getState() != AudioRecord.STATE_INITIALIZED) {
                    throw new Exception("AudioRecord initialization failed");
                }
                audioRecorder.setRecordPositionUpdateListener(updateListener);
                audioRecorder.setPositionNotificationPeriod(mPeriodInFrames);
                filePath = null;
                state = State.INITIALIZING;
            } catch (Exception e) {
                if (e.getMessage() != null) {
                    Log.e(WavAudioRecorder.class.getName(), e.getMessage());
                } else {
                    Log.e(WavAudioRecorder.class.getName(), "Unknown error occurred while initializing recording");
                }
                state = State.ERROR;
            }
        }

        /**
         * Sets output file path, call directly after construction/reset.
         *
         * @param argPath file path
         */
        public void setOutputFile(String argPath) {
            try {
                if (state == State.INITIALIZING) {
                    filePath = argPath;
                }
            } catch (Exception e) {
                if (e.getMessage() != null) {
                    Log.e(WavAudioRecorder.class.getName(), e.getMessage());
                } else {
                    Log.e(WavAudioRecorder.class.getName(), "Unknown error occured while setting output path");
                }
                state = State.ERROR;
            }
        }


        /**
         * Prepares the recorder for recording, in case the recorder is not in the INITIALIZING state and the file path was not set
         * the recorder is set to the ERROR state, which makes a reconstruction necessary.
         * In case uncompressed recording is toggled, the header of the wave file is written.
         * In case of an exception, the state is changed to ERROR
         */
        public void prepare() {
            try {
                if (state == State.INITIALIZING) {
                    if ((audioRecorder.getState() == AudioRecord.STATE_INITIALIZED) & (filePath != null)) {
                        // write file header
                        // If the directory does not exist, create it
                        if (!rootDir.exists()) {
                            rootDir.mkdirs(); // Create necessary directories
                        }
                        randomAccessWriter = new RandomAccessFile(filePath, "rw");
                        randomAccessWriter.setLength(0); // Set file length to 0, to prevent unexpected behavior in case the file already existed
                        randomAccessWriter.writeBytes("RIFF");
                        randomAccessWriter.writeInt(0); // Final file size not known yet, write 0
                        randomAccessWriter.writeBytes("WAVE");
                        randomAccessWriter.writeBytes("fmt ");
                        randomAccessWriter.writeInt(Integer.reverseBytes(16)); // Sub-chunk size, 16 for PCM
                        randomAccessWriter.writeShort(Short.reverseBytes((short) 1)); // AudioFormat, 1 for PCM
                        randomAccessWriter.writeShort(Short.reverseBytes(nChannels));// Number of channels, 1 for mono, 2 for stereo
                        randomAccessWriter.writeInt(Integer.reverseBytes(sRate)); // Sample rate
                        randomAccessWriter.writeInt(Integer.reverseBytes(sRate * nChannels * mBitsPersample / 8)); // Byte rate, SampleRate*NumberOfChannels*mBitsPersample/8
                        randomAccessWriter.writeShort(Short.reverseBytes((short) (nChannels * mBitsPersample / 8))); // Block align, NumberOfChannels*mBitsPersample/8
                        randomAccessWriter.writeShort(Short.reverseBytes(mBitsPersample)); // Bits per sample
                        randomAccessWriter.writeBytes("data");
                        randomAccessWriter.writeInt(0); // Data chunk size not known yet, write 0
                        buffer = new byte[mPeriodInFrames * mBitsPersample / 8 * nChannels];
                        state = State.READY;
                    } else {
                        Log.e(WavAudioRecorder.class.getName(), "prepare() method called on uninitialized recorder");
                        state = State.ERROR;
                    }
                } else {
                    Log.e(WavAudioRecorder.class.getName(), "prepare() method called on illegal state");
                    release();
                    state = State.ERROR;
                }
            } catch (Exception e) {
                if (e.getMessage() != null) {
                    Log.e(WavAudioRecorder.class.getName(), e.getMessage());
                } else {
                    Log.e(WavAudioRecorder.class.getName(), "Unknown error occured in prepare()");
                }
                state = State.ERROR;
            }
        }

        /**
         * Releases the resources associated with this class, and removes the unnecessary files, when necessary
         */
        public void release() {
            if (state == State.RECORDING) {
                stop();
            } else {
                if (state == State.READY) {
                    try {
                        randomAccessWriter.close();
                    } catch (IOException e) {
                        Log.e(WavAudioRecorder.class.getName(), "I/O exception occured while closing output file");
                    }
//                    (new File(filePath)).delete();
                }
            }

            if (audioRecorder != null) {
                audioRecorder.release();
            }
        }

        /**
         * Resets the recorder to the INITIALIZING state, as if it was just created.
         * In case the class was in RECORDING state, the recording is stopped.
         * In case of exceptions the class is set to the ERROR state.
         */
        public void reset(Context context) {
            try {
                if (state != State.ERROR) {
                    release();
                    filePath = null; // Reset file path
                    if (ActivityCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                        // TODO: Consider calling
                        //    ActivityCompat#requestPermissions
                        // here to request the missing permissions, and then overriding
                        //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                        //                                          int[] grantResults)
                        // to handle the case where the user grants the permission. See the documentation
                        // for ActivityCompat#requestPermissions for more details.
                        return;
                    }
                    audioRecorder = new AudioRecord(mAudioSource, sRate, nChannels, aFormat, mBufferSize);
                    if (audioRecorder.getState() != AudioRecord.STATE_INITIALIZED) {
                        throw new Exception("AudioRecord initialization failed");
                    }
                    audioRecorder.setRecordPositionUpdateListener(updateListener);
                    audioRecorder.setPositionNotificationPeriod(mPeriodInFrames);
                    state = State.INITIALIZING;
                }
            } catch (Exception e) {
                Log.e(WavAudioRecorder.class.getName(), e.getMessage());
                state = State.ERROR;
            }
        }

        /**
         * Starts the recording, and sets the state to RECORDING.
         * Call after prepare().
         */
        public void start() {
            if (state == State.READY) {
                payloadSize = 0;
                audioRecorder.startRecording();
                audioRecorder.read(buffer, 0, buffer.length);    //[TODO: is this necessary]read the existing data in audio hardware, but don't do anything
                state = State.RECORDING;
            } else {
                Log.e(WavAudioRecorder.class.getName(), "start() called on illegal state");
                state = State.ERROR;
            }
        }

        /**
         * Stops the recording, and sets the state to STOPPED.
         * In case of further usage, a reset is needed.
         * Also finalizes the wave file in case of uncompressed recording.
         */
        public void stop() {
            if (state == State.RECORDING) {
                audioRecorder.stop();
                try {
                    randomAccessWriter.seek(4); // Write size to RIFF header
                    randomAccessWriter.writeInt(Integer.reverseBytes(36 + payloadSize));

                    randomAccessWriter.seek(40); // Write size to Subchunk2Size field
                    randomAccessWriter.writeInt(Integer.reverseBytes(payloadSize));

//                    randomAccessWriter.close();
                } catch (IOException e) {
                    Log.e(WavAudioRecorder.class.getName(), "I/O exception occurred while closing output file");
                    state = State.ERROR;
                }
                state = State.STOPPED;
            } else {
                Log.e(WavAudioRecorder.class.getName(), "stop() called on illegal state");
                state = State.ERROR;
            }
        }
    }
}