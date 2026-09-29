package com.alex.aivideo;

import java.nio.FloatBuffer;

/**
 * Converts one frame at a time from decoder NCTHW float output
 * [1,3,T,H,W] in [-1,1] into RGBA8.
 */
public final class VideoTensorFrames {
    private VideoTensorFrames() {}

    public static byte[] readRgbaFrame(
            FloatBuffer video,
            int frameIndex,
            int frames,
            int width,
            int height
    ) {
        if (video == null) {
            throw new IllegalArgumentException("video buffer=null");
        }
        if (frameIndex < 0 || frameIndex >= frames) {
            throw new IllegalArgumentException("Неверный frameIndex.");
        }

        int spatial = width * height;
        int expected = 3 * frames * spatial;
        if (video.capacity() < expected) {
            throw new IllegalArgumentException(
                    "Decoder output слишком мал: "
                            + video.capacity() + ", ожидалось >= " + expected
            );
        }

        byte[] rgba = new byte[spatial * 4];
        int rBase = frameIndex * spatial;
        int gBase = frames * spatial + frameIndex * spatial;
        int bBase = 2 * frames * spatial + frameIndex * spatial;

        for (int i = 0; i < spatial; i++) {
            int p = i * 4;
            rgba[p] = normalizedToByte(video.get(rBase + i));
            rgba[p + 1] = normalizedToByte(video.get(gBase + i));
            rgba[p + 2] = normalizedToByte(video.get(bBase + i));
            rgba[p + 3] = (byte) 0xff;
        }

        return rgba;
    }

    static byte normalizedToByte(float value) {
        float clamped = Math.max(-1.0f, Math.min(1.0f, value));
        int v = Math.round((clamped + 1.0f) * 127.5f);
        return (byte) Math.max(0, Math.min(255, v));
    }
}
