package com.whisper.android.tflitecpp.utils;

import android.os.Handler;
import android.os.Looper;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ModelDownloader {
    private static final ExecutorService executor = Executors.newSingleThreadExecutor();
    private static final Handler mainHandler = new Handler(Looper.getMainLooper());

    public interface Callback {
        void onProgress(int percent);
        void onComplete(File file);
        void onError(String error);
    }

    public static void download(String downloadUrl, String fileName, File destDir, Callback callback) {
        executor.execute(() -> {
            try {
                URL url = new URL(downloadUrl);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setInstanceFollowRedirects(true);
                conn.connect();

                int total = conn.getContentLength();
                File outFile = new File(destDir, fileName);
                File tmpFile = new File(destDir, fileName + ".tmp");

                try (InputStream in = conn.getInputStream();
                     FileOutputStream out = new FileOutputStream(tmpFile)) {
                    byte[] buf = new byte[8192];
                    long downloaded = 0;
                    int n;
                    while ((n = in.read(buf)) != -1) {
                        out.write(buf, 0, n);
                        downloaded += n;
                        if (total > 0) {
                            int pct = (int) (downloaded * 100 / total);
                            mainHandler.post(() -> callback.onProgress(pct));
                        }
                    }
                }

                tmpFile.renameTo(outFile);
                mainHandler.post(() -> callback.onComplete(outFile));
            } catch (Exception e) {
                mainHandler.post(() -> callback.onError(e.getMessage()));
            }
        });
    }
}
