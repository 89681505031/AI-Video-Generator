package com.alex.aivideo;

import android.content.Context;
import android.os.Environment;
import android.os.StatFs;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

public final class ModelRelease {
    public static final String INSTALL_MARKER = ".model_release_sha256";
    public static final String FILE_NAME =
            "mobile_i2v_v2_512_base30_turbo.zip";

    public static final String DOWNLOAD_URL =
            "https://github.com/89681505031/AI-Video-Generator/"
                    + "releases/download/mobile-model-v0.2/"
                    + FILE_NAME;

    public static final String SHA256 =
            "f976263fe16a3a9f913cd7fd6b76cb31efbd99c04ce6c9b8d589c7b6c270885d";

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

    public static boolean isCurrentInstalled(File modelDir) {
        File marker = new File(modelDir, INSTALL_MARKER);
        if (!marker.isFile()) {
            return false;
        }
        try {
            byte[] data = new byte[(int) marker.length()];
            try (FileInputStream in = new FileInputStream(marker)) {
                int offset = 0;
                while (offset < data.length) {
                    int read = in.read(data, offset, data.length - offset);
                    if (read < 0) break;
                    offset += read;
                }
                if (offset != data.length) return false;
            }
            String value = new String(data, StandardCharsets.UTF_8).trim();
            return SHA256.equals(value);
        } catch (Exception ignored) {
            return false;
        }
    }

    public static void markInstalled(File modelDir) throws Exception {
        if (modelDir == null || !modelDir.isDirectory()) {
            throw new IllegalArgumentException("Папка установленной модели не найдена.");
        }
        File marker = new File(modelDir, INSTALL_MARKER);
        try (FileOutputStream out = new FileOutputStream(marker, false)) {
            out.write((SHA256 + "\n").getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
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
