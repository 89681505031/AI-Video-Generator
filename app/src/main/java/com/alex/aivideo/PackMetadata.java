package com.alex.aivideo;

import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;

public final class PackMetadata {
    public final String variant;
    public final int width;
    public final int height;
    public final int frames;
    public final int latentChannels;
    public final int vaeDownsampleRate;
    public final int temporalLatents;
    public final int samplingSteps;
    public final boolean textConditioning;
    public final int textMaxLength;
    public final int captionChannels;
    public final String sourceCommit;

    private PackMetadata(
            String variant,
            int width,
            int height,
            int frames,
            int latentChannels,
            int vaeDownsampleRate,
            int temporalLatents,
            int samplingSteps,
            boolean textConditioning,
            int textMaxLength,
            int captionChannels,
            String sourceCommit
    ) {
        this.variant = variant;
        this.width = width;
        this.height = height;
        this.frames = frames;
        this.latentChannels = latentChannels;
        this.vaeDownsampleRate = vaeDownsampleRate;
        this.temporalLatents = temporalLatents;
        this.samplingSteps = samplingSteps;
        this.textConditioning = textConditioning;
        this.textMaxLength = textMaxLength;
        this.captionChannels = captionChannels;
        this.sourceCommit = sourceCommit;
    }

    public static PackMetadata load(File dir) throws Exception {
        File runtime = new File(dir, "runtime.json");
        if (!runtime.isFile()) {
            throw new IllegalArgumentException("В model pack нет runtime.json.");
        }

        byte[] bytes = readAll(runtime, 1024 * 1024);
        JSONObject json = new JSONObject(new String(bytes, StandardCharsets.UTF_8));

        String variant = json.optString("variant", "").trim().toLowerCase();
        int width = json.optInt("width", 0);
        int height = json.optInt("height", 0);
        int frames = json.optInt("frames", 0);
        int latentChannels = json.optInt("latent_channels", 0);
        int vaeDownsampleRate = json.optInt("vae_downsample_rate", 0);
        int temporalLatents = json.optInt("temporal_latents", 0);
        int samplingSteps = json.optInt("sampling_steps", 0);
        boolean textConditioning = json.optBoolean("text_conditioning", false);
        int textMaxLength = json.optInt("text_max_length", 0);
        int captionChannels = json.optInt("caption_channels", 0);
        String sourceCommit = json.optString("source_commit", "").trim();

        if (!variant.equals("base") && !variant.equals("distilled")) {
            throw new IllegalArgumentException("variant должен быть base или distilled.");
        }
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Размер кадра должен быть положительным.");
        }
        if ((width & 1) != 0 || (height & 1) != 0) {
            throw new IllegalArgumentException("Ширина и высота видео должны быть чётными.");
        }
        if (frames != MobileI2VContract.OUTPUT_FRAMES) {
            throw new IllegalArgumentException(
                    "MobileI2V v1 ожидает " + MobileI2VContract.OUTPUT_FRAMES + " кадров."
            );
        }
        if (latentChannels != MobileI2VContract.LATENT_CHANNELS) {
            throw new IllegalArgumentException(
                    "Ожидается " + MobileI2VContract.LATENT_CHANNELS + " latent-каналов."
            );
        }
        if (vaeDownsampleRate != MobileI2VContract.SPATIAL_DOWNSAMPLE) {
            throw new IllegalArgumentException(
                    "Ожидается VAE downsample ×" + MobileI2VContract.SPATIAL_DOWNSAMPLE + "."
            );
        }
        if (temporalLatents != MobileI2VContract.LATENT_TIME) {
            throw new IllegalArgumentException("Неверное число temporal latents.");
        }
        if (samplingSteps <= 0 || samplingSteps > 60) {
            throw new IllegalArgumentException("Некорректное число sampling_steps.");
        }
        if (variant.equals("distilled") && samplingSteps > 4) {
            throw new IllegalArgumentException("Distilled pack не должен объявлять больше 4 шагов.");
        }
        if (textConditioning
                && (textMaxLength != MobileI2VContract.TEXT_MAX_TOKENS
                || captionChannels != MobileI2VContract.TEXT_CHANNELS)) {
            throw new IllegalArgumentException(
                    "Text pack должен использовать Qwen2 max_length="
                            + MobileI2VContract.TEXT_MAX_TOKENS
                            + " и " + MobileI2VContract.TEXT_CHANNELS + " каналов."
            );
        }
        if (sourceCommit.isEmpty()) {
            throw new IllegalArgumentException("runtime.json должен содержать source_commit.");
        }

        File nullCondition = new File(dir, "null_condition.bin");
        long expectedNullBytes =
                1L * 1L * MobileI2VContract.TEXT_MAX_TOKENS
                        * MobileI2VContract.TEXT_CHANNELS * 2L;
        if (!nullCondition.isFile() || nullCondition.length() != expectedNullBytes) {
            throw new IllegalArgumentException(
                    "null_condition.bin должен быть FP16 [1,1,"
                            + MobileI2VContract.TEXT_MAX_TOKENS + ","
                            + MobileI2VContract.TEXT_CHANNELS + "] ("
                            + expectedNullBytes + " байт)."
            );
        }

        File nullMask = new File(dir, "null_attention_mask.bin");
        byte[] maskBytes = readAll(
                nullMask,
                MobileI2VContract.TEXT_MAX_TOKENS
        );
        if (maskBytes.length != MobileI2VContract.TEXT_MAX_TOKENS) {
            throw new IllegalArgumentException(
                    "null_attention_mask.bin должен содержать ровно "
                            + MobileI2VContract.TEXT_MAX_TOKENS + " байт."
            );
        }
        for (byte value : maskBytes) {
            int unsigned = value & 0xff;
            if (unsigned != 0 && unsigned != 1) {
                throw new IllegalArgumentException(
                        "null_attention_mask.bin должен содержать только 0/1."
                );
            }
        }
        return new PackMetadata(
                variant, width, height, frames, latentChannels, vaeDownsampleRate,
                temporalLatents, samplingSteps, textConditioning, textMaxLength,
                captionChannels, sourceCommit
        );
    }

    public int latentWidth() {
        return MobileI2VContract.latentWidth(width);
    }

    public int latentHeight() {
        return MobileI2VContract.latentHeight(height);
    }

    public int sequencePositions() {
        return MobileI2VContract.sequencePositions(width, height);
    }

    public String summary() {
        String speed = variant.equals("distilled")
                ? "DISTILLED • " + samplingSteps + " шага"
                : "BASE • " + samplingSteps + " шагов (медленнее)";
        String prompt = textConditioning ? "текст: да" : "текст: Lite/без Qwen2";
        return speed + "\n"
                + width + "×" + height + " • " + frames + " кадров • " + prompt
                + "\nlatent: " + latentChannels + "×" + temporalLatents + "×"
                + latentHeight() + "×" + latentWidth()
                + " • positions=" + sequencePositions();
    }

    private static byte[] readAll(File file, int maxBytes) throws Exception {
        long length = file.length();
        if (length < 0 || length > maxBytes) {
            throw new IllegalArgumentException("Файл метаданных слишком большой.");
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
                throw new IllegalStateException("Не удалось полностью прочитать метаданные.");
            }
        }
        return data;
    }
}
