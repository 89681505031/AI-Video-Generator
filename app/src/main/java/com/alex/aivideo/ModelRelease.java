package com.alex.aivideo;

import android.content.Context;
import android.os.Environment;
import android.os.StatFs;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.security.MessageDigest;

public final class ModelRelease {
    public static final String FILE_NAME =
            "mobile_i2v_v1_512_base30_turbo.zip";

    public static final String DOWNLOAD_URL =
            "https://github.com/89681505031/AI-Video-Generator/"
                    + "releases/download/mobile-model-v0.1/"
                    + FILE_NAME;

    public static final String SHA256 =
            "b293b0d49d387b24d9b9e791f341df418048bdb89a177b42f05191295309c02b";

    public static final long REQUIRED_FREE_BYTES =
            5L * 1024L * 1024L * 1024L;

    private ModelRelease() {}

    public static File downloadFile(Context context) {
        File root = context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS);
        if (root == null) {
            throw new IllegalStateException(
                    "Android не предоставил папку загрузок приложения."
            );
        }
        if (!root.exists() && !root.mkdirs()) {
            throw new IllegalStateException(
                    "Не удалось создать папку загрузок модели."
            );
        }
        return new File(root, FILE_NAME);
    }

    public static void requireEnoughSpace(Context context) {
        StatFs internal = new StatFs(context.getFilesDir().getAbsolutePath());
        File download = downloadFile(context);
        StatFs external = new StatFs(
                download.getParentFile().getAbsolutePath()
        );

        if (internal.getAvailableBytes() < REQUIRED_FREE_BYTES
                || external.getAvailableBytes() < REQUIRED_FREE_BYTES) {
            throw new IllegalStateException(
                    "Для модели нужно минимум 5 ГБ свободного места."
            );
        }
    }

    public static void verifyDownloadedZip(File file) throws Exception {
        if (file == null || !file.isFile()) {
            throw new IllegalArgumentException(
                    "Скачанный ZIP модели не найден."
            );
        }

        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] buffer = new byte[1024 * 1024];

        try (BufferedInputStream in =
                     new BufferedInputStream(new FileInputStream(file))) {
            int read;
            while ((read = in.read(buffer)) != -1) {
                digest.update(buffer, 0, read);
            }
        }

        StringBuilder actual = new StringBuilder(64);
        for (byte b : digest.digest()) {
            actual.append(String.format("%02x", b & 0xff));
        }

        if (!SHA256.equals(actual.toString())) {
            throw new SecurityException(
                    "SHA-256 скачанного model pack не совпал."
            );
        }
    }
}
