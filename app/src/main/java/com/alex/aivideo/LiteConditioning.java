package com.alex.aivideo;

import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;

/**
 * Precomputed empty-prompt Qwen2 conditioning used by the phone Lite profile.
 */
public final class LiteConditioning {
    public final float[] embeddings;
    public final long[] attentionMask;

    private LiteConditioning(float[] embeddings, long[] attentionMask) {
        this.embeddings = embeddings;
        this.attentionMask = attentionMask;
    }

    public static LiteConditioning load(File modelDirectory) throws Exception {
        File embeddingFile = new File(modelDirectory, "null_condition.bin");
        File maskFile = new File(modelDirectory, "null_attention_mask.bin");

        int embeddingElements =
                MobileI2VContract.TEXT_MAX_TOKENS * MobileI2VContract.TEXT_CHANNELS;
        byte[] embeddingBytes = readExactly(
                embeddingFile,
                embeddingElements * 2
        );

        ShortBuffer half = ByteBuffer.wrap(embeddingBytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asShortBuffer();

        float[] embeddings = new float[embeddingElements];
        for (int i = 0; i < embeddings.length; i++) {
            embeddings[i] = Fp16.toFloat(half.get(i));
        }

        byte[] maskBytes = readExactly(
                maskFile,
                MobileI2VContract.TEXT_MAX_TOKENS
        );
        long[] mask = new long[maskBytes.length];
        for (int i = 0; i < maskBytes.length; i++) {
            int value = maskBytes[i] & 0xff;
            if (value != 0 && value != 1) {
                throw new IllegalArgumentException(
                        "Lite attention mask должен содержать только 0/1."
                );
            }
            mask[i] = value;
        }

        return new LiteConditioning(embeddings, mask);
    }

    private static byte[] readExactly(File file, int expectedBytes) throws Exception {
        if (!file.isFile() || file.length() != expectedBytes) {
            throw new IllegalArgumentException(
                    "Неверный размер файла " + file.getName()
                            + ": ожидалось " + expectedBytes + " байт."
            );
        }

        byte[] bytes = new byte[expectedBytes];
        try (FileInputStream in = new FileInputStream(file)) {
            int offset = 0;
            while (offset < bytes.length) {
                int read = in.read(bytes, offset, bytes.length - offset);
                if (read < 0) break;
                offset += read;
            }
            if (offset != bytes.length) {
                throw new IllegalStateException(
                        "Не удалось полностью прочитать " + file.getName()
                );
            }
        }
        return bytes;
    }
}
