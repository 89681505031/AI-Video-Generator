package com.alex.aivideo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;

import java.io.InputStream;

/**
 * Prepares the selected source image exactly as MobileI2V expects:
 * RGB, resized to the runtime profile and normalized from [0,255] to [-1,1].
 *
 * Output layout is contiguous NCTHW with N=1 and T=1:
 * [1, 3, 1, height, width].
 */
public final class ImageTensorPreprocessor {
    private ImageTensorPreprocessor() {}

    public static float[] loadNcthw(
            Context context,
            Uri uri,
            int targetWidth,
            int targetHeight
    ) throws Exception {
        if (context == null || uri == null) {
            throw new IllegalArgumentException("context/uri не должны быть null.");
        }
        if (targetWidth <= 0 || targetHeight <= 0) {
            throw new IllegalArgumentException("Неверный целевой размер изображения.");
        }

        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream in = context.getContentResolver().openInputStream(uri)) {
            if (in == null) {
                throw new IllegalArgumentException("Не удалось открыть изображение.");
            }
            BitmapFactory.decodeStream(in, null, bounds);
        }

        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw new IllegalArgumentException("Не удалось определить размер изображения.");
        }

        BitmapFactory.Options decode = new BitmapFactory.Options();
        decode.inPreferredConfig = Bitmap.Config.ARGB_8888;
        decode.inSampleSize = chooseSampleSize(
                bounds.outWidth,
                bounds.outHeight,
                targetWidth,
                targetHeight
        );

        Bitmap source;
        try (InputStream in = context.getContentResolver().openInputStream(uri)) {
            if (in == null) {
                throw new IllegalArgumentException("Не удалось повторно открыть изображение.");
            }
            source = BitmapFactory.decodeStream(in, null, decode);
        }

        if (source == null) {
            throw new IllegalArgumentException("Android не смог декодировать изображение.");
        }

        Bitmap scaled = source;
        try {
            if (source.getWidth() != targetWidth || source.getHeight() != targetHeight) {
                scaled = Bitmap.createScaledBitmap(
                        source,
                        targetWidth,
                        targetHeight,
                        true
                );
            }
            return bitmapToNcthw(scaled);
        } finally {
            if (scaled != source && !scaled.isRecycled()) {
                scaled.recycle();
            }
            if (!source.isRecycled()) {
                source.recycle();
            }
        }
    }

    static float[] bitmapToNcthw(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int spatial = width * height;
        float[] tensor = new float[3 * spatial];
        int[] row = new int[width];

        for (int y = 0; y < height; y++) {
            bitmap.getPixels(row, 0, width, 0, y, width, 1);
            int rowOffset = y * width;

            for (int x = 0; x < width; x++) {
                int argb = row[x];
                int index = rowOffset + x;

                tensor[index] = normalize((argb >> 16) & 0xff);
                tensor[spatial + index] = normalize((argb >> 8) & 0xff);
                tensor[2 * spatial + index] = normalize(argb & 0xff);
            }
        }

        return tensor;
    }

    private static float normalize(int channel) {
        return channel / 127.5f - 1.0f;
    }

    private static int chooseSampleSize(
            int sourceWidth,
            int sourceHeight,
            int targetWidth,
            int targetHeight
    ) {
        int sample = 1;
        while (sourceWidth / (sample * 2) >= targetWidth
                && sourceHeight / (sample * 2) >= targetHeight) {
            sample *= 2;
        }
        return sample;
    }
}
