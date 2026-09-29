package com.alex.aivideo;

import android.content.Context;
import android.net.Uri;

import java.io.File;

import ai.onnxruntime.OrtEnvironment;

public final class MobileVideoEngine {
    public interface ProgressListener {
        void onProgress(int percent, String stage);
    }

    private final Context context;

    public MobileVideoEngine(Context context) {
        this.context = context.getApplicationContext();
    }

    public boolean runtimeReady() {
        try {
            OrtEnvironment.getEnvironment();
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    public File modelDirectory() {
        return new File(
                context.getFilesDir(),
                "models/" + ModelPackInstaller.PACK_ID
        );
    }

    public PackMetadata metadataOrNull() {
        try {
            if (!ModelRelease.isCurrentInstalled(modelDirectory())) {
                return null;
            }
            return ModelPackInstaller.validatePack(modelDirectory());
        } catch (Exception ignored) {
            return null;
        }
    }

    public String modelPackStatus() {
        File dir = modelDirectory();
        if (!dir.exists()) {
            return "MobileI2V model pack ещё не установлен.";
        }

        try {
            PackMetadata metadata = ModelPackInstaller.validatePack(dir);
            if (!ModelRelease.isCurrentInstalled(dir)) {
                return "MobileI2V pack устарел. Скачайте исправленную модель v0.2.";
            }
            return "MobileI2V pack v0.2: OK\n" + metadata.summary();
        } catch (Exception e) {
            return "Model pack не прошёл проверку:\n" + e.getMessage();
        }
    }

    public boolean modelPackReady() {
        return metadataOrNull() != null;
    }

    public boolean supportsPrompt() {
        return false;
    }

    public String deepModelCheck() throws Exception {
        if (!modelPackReady()) {
            throw new IllegalStateException("Model pack ещё не установлен.");
        }
        return new StagedOrtRunner(modelDirectory()).validateModelFiles();
    }

    public File generate(
            Uri sourceImage,
            ProgressListener progress
    ) throws Exception {
        if (!ModelRelease.isCurrentInstalled(modelDirectory())) {
            throw new IllegalStateException(
                    "Установленная модель устарела. Скачайте MobileI2V v0.2."
            );
        }
        PackMetadata metadata = ModelPackInstaller.validatePack(
                modelDirectory()
        );

        if (sourceImage == null) {
            throw new IllegalArgumentException(
                    "Исходное изображение не выбрано."
            );
        }

        // v4 decoder output is a full FP32 NCTHW tensor in native ORT memory.
        // Restrict the first phone-ready implementation to 512-class outputs
        // so a 720p decoder cannot unexpectedly reserve ~188 MB just for output.
        if (metadata.width * metadata.height > 512 * 512) {
            throw new IllegalStateException(
                    "Этот pack слишком тяжёлый для первого мобильного decoder path. "
                            + "Нужен 512×512 mobile pack."
            );
        }

        emit(progress, 2, "Подготовка изображения");
        float[] image = ImageTensorPreprocessor.loadNcthw(
                context,
                sourceImage,
                metadata.width,
                metadata.height
        );

        long seed = System.nanoTime();
        MobileI2VOrtCore core = new MobileI2VOrtCore(modelDirectory());

        emit(progress, 10, "VAE encode");
        float[] guide = core.encodeGuideImage(
                image,
                metadata,
                seed
        );
        image = null;

        emit(progress, 20, "Создание latent");
        float[] latent = LatentInitializer.createConditionedNoise(
                metadata.width,
                metadata.height,
                seed ^ 0x5DEECE66DL,
                guide
        );

        emit(progress, 25, "MobileI2V");
        latent = core.denoise(
                latent,
                guide,
                metadata,
                (step, total) -> {
                    int pct = 25 + Math.round(
                            55.0f * (step + 1) / Math.max(1, total)
                    );
                    emit(
                            progress,
                            pct,
                            "MobileI2V шаг "
                                    + (step + 1)
                                    + "/"
                                    + total
                    );
                }
        );
        guide = null;

        File output = newOutputFile();
        emit(progress, 82, "Декодирование видео");

        core.decodeToMp4(
                latent,
                metadata,
                output,
                (frame, total) -> {
                    int pct = 82 + Math.round(
                            17.0f * frame / Math.max(1, total)
                    );
                    emit(
                            progress,
                            Math.min(99, pct),
                            "MP4 кадр "
                                    + frame
                                    + "/"
                                    + total
                    );
                }
        );

        emit(progress, 100, "Готово");
        return output;
    }

    public File newOutputFile() {
        File base = context.getExternalFilesDir(null);
        if (base == null) {
            base = context.getFilesDir();
        }

        File dir = new File(base, "generated");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException(
                    "Не удалось создать папку generated."
            );
        }

        return new File(
                dir,
                "mobilei2v_" + System.currentTimeMillis() + ".mp4"
        );
    }

    private static void emit(
            ProgressListener listener,
            int percent,
            String stage
    ) {
        if (listener != null) {
            listener.onProgress(
                    Math.max(0, Math.min(100, percent)),
                    stage
            );
        }
    }
}
