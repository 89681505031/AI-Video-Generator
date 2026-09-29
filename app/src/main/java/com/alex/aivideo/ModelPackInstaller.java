package com.alex.aivideo;

import android.content.Context;
import android.net.Uri;

import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Iterator;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class ModelPackInstaller {
    public static final String PACK_ID = "mobile_i2v_v1";
    public static final int FORMAT_VERSION = 4;

    public static final String[] CORE_FILES = {
            "vae_encoder.onnx",
            "mobilei2v_transformer.onnx",
            "video_decoder.onnx",
            "runtime.json",
            "manifest.json"
    };

    private ModelPackInstaller() {}

    public static File install(Context context, Uri zipUri) throws Exception {
        File modelsRoot = new File(context.getFilesDir(), "models");
        if (!modelsRoot.exists() && !modelsRoot.mkdirs()) {
            throw new IllegalStateException("Не удалось создать папку моделей.");
        }

        File target = new File(modelsRoot, PACK_ID);
        File temp = new File(modelsRoot, "." + PACK_ID + ".installing");
        File backup = new File(modelsRoot, "." + PACK_ID + ".old");

        deleteRecursively(temp);
        deleteRecursively(backup);
        if (!temp.mkdirs()) {
            throw new IllegalStateException("Не удалось создать временную папку.");
        }

        try {
            extractZipSafely(context, zipUri, temp);
            validatePack(temp);

            if (target.exists() && !target.renameTo(backup)) {
                throw new IllegalStateException("Не удалось подготовить замену старой модели.");
            }

            if (!temp.renameTo(target)) {
                if (backup.exists()) {
                    //noinspection ResultOfMethodCallIgnored
                    backup.renameTo(target);
                }
                throw new IllegalStateException("Не удалось активировать новый model pack.");
            }

            deleteRecursively(backup);
            return target;
        } catch (Exception e) {
            deleteRecursively(temp);
            throw e;
        }
    }

    public static PackMetadata validatePack(File dir) throws Exception {
        File manifestFile = new File(dir, "manifest.json");
        if (!manifestFile.isFile()) {
            throw new IllegalArgumentException("В model pack нет manifest.json.");
        }

        JSONObject manifest = new JSONObject(readUtf8(manifestFile));
        if (!PACK_ID.equals(manifest.optString("pack_id"))) {
            throw new IllegalArgumentException("Неверный pack_id. Нужен " + PACK_ID + ".");
        }
        if (manifest.optInt("format_version", -1) != FORMAT_VERSION) {
            throw new IllegalArgumentException(
                    "Нужен format_version=" + FORMAT_VERSION + "."
            );
        }

        JSONObject files = manifest.optJSONObject("files");
        if (files == null || files.length() == 0) {
            throw new IllegalArgumentException("В manifest.json нет объекта files.");
        }

        for (String name : CORE_FILES) {
            if (!new File(dir, name).isFile()) {
                throw new IllegalArgumentException(
                        "Не хватает обязательного файла: " + name
                );
            }
        }

        PackMetadata metadata = PackMetadata.load(dir);

        Iterator<String> keys = files.keys();
        while (keys.hasNext()) {
            String name = keys.next();
            if (name.contains("/") || name.contains("\\") || name.equals("manifest.json")) {
                throw new SecurityException("Недопустимое имя в manifest: " + name);
            }

            File file = new File(dir, name);
            if (!file.isFile()) {
                throw new IllegalArgumentException(
                        "Manifest ссылается на отсутствующий файл: " + name
                );
            }

            String expected = files.optString(name, "").trim().toLowerCase();
            if (expected.length() != 64) {
                throw new IllegalArgumentException(
                        "Нет корректного SHA-256 для: " + name
                );
            }

            String actual = sha256(file);
            if (!expected.equals(actual)) {
                throw new SecurityException("SHA-256 не совпал: " + name);
            }
        }

        for (String name : CORE_FILES) {
            if (name.equals("manifest.json")) continue;
            if (!files.has(name)) {
                throw new IllegalArgumentException(
                        "Manifest не содержит SHA-256 для: " + name
                );
            }
        }

        return metadata;
    }

    private static void extractZipSafely(
            Context context,
            Uri uri,
            File destination
    ) throws Exception {
        String root = destination.getCanonicalPath() + File.separator;

        try (InputStream raw = context.getContentResolver().openInputStream(uri)) {
            if (raw == null) {
                throw new IllegalArgumentException("Не удалось открыть ZIP.");
            }

            try (ZipInputStream zip = new ZipInputStream(new BufferedInputStream(raw))) {
                ZipEntry entry;
                byte[] buffer = new byte[1024 * 1024];

                while ((entry = zip.getNextEntry()) != null) {
                    String name = entry.getName().replace('\\', '/');
                    if (name.startsWith("/") || name.contains("../")) {
                        throw new SecurityException("Небезопасный путь в ZIP: " + name);
                    }

                    File out = new File(destination, name);
                    String canonical = out.getCanonicalPath();
                    if (!canonical.startsWith(root)) {
                        throw new SecurityException(
                                "ZIP пытается выйти за папку model pack."
                        );
                    }

                    if (entry.isDirectory()) {
                        if (!out.exists() && !out.mkdirs()) {
                            throw new IllegalStateException(
                                    "Не удалось создать папку: " + name
                            );
                        }
                    } else {
                        File parent = out.getParentFile();
                        if (parent != null && !parent.exists() && !parent.mkdirs()) {
                            throw new IllegalStateException(
                                    "Не удалось создать папку для: " + name
                            );
                        }

                        try (BufferedOutputStream os =
                                     new BufferedOutputStream(new FileOutputStream(out))) {
                            int read;
                            while ((read = zip.read(buffer)) != -1) {
                                os.write(buffer, 0, read);
                            }
                        }
                    }
                    zip.closeEntry();
                }
            }
        }
    }

    private static String readUtf8(File file) throws Exception {
        long length = file.length();
        if (length > 1024 * 1024) {
            throw new IllegalArgumentException("manifest.json слишком большой.");
        }

        byte[] data = new byte[(int) length];
        try (FileInputStream in = new FileInputStream(file)) {
            int offset = 0;
            while (offset < data.length) {
                int read = in.read(data, offset, data.length - offset);
                if (read < 0) break;
                offset += read;
            }
            if (offset != data.length) {
                throw new IllegalStateException(
                        "Не удалось полностью прочитать manifest.json."
                );
            }
        }
        return new String(data, StandardCharsets.UTF_8);
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] buffer = new byte[1024 * 1024];

        try (BufferedInputStream in =
                     new BufferedInputStream(new FileInputStream(file))) {
            int read;
            while ((read = in.read(buffer)) != -1) {
                digest.update(buffer, 0, read);
            }
        }

        StringBuilder hex = new StringBuilder(64);
        for (byte b : digest.digest()) {
            hex.append(String.format("%02x", b & 0xff));
        }
        return hex.toString();
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) return;

        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }

        //noinspection ResultOfMethodCallIgnored
        file.delete();
    }
}
