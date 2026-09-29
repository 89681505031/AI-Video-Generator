package com.alex.aivideo;

import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMuxer;

import java.io.File;
import java.nio.ByteBuffer;

/**
 * Streaming H.264/MP4 writer for RGBA frames.
 *
 * Frames are converted to YUV420 and submitted one at a time so the complete
 * clip does not have to live in Java memory before encoding.
 */
public final class H264Mp4Encoder implements AutoCloseable {
    private static final String MIME = MediaFormat.MIMETYPE_VIDEO_AVC;
    private static final long TIMEOUT_US = 10_000;

    private final int width;
    private final int height;
    private final int fps;
    private final MediaCodec codec;
    private final MediaMuxer muxer;
    private final MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
    private final byte[] yuvFrame;
    private final boolean semiPlanar;

    private boolean muxerStarted;
    private boolean finished;
    private int trackIndex = -1;
    private long frameIndex;

    public H264Mp4Encoder(
            File output,
            int width,
            int height,
            int fps,
            int bitrate
    ) throws Exception {
        if ((width & 1) != 0 || (height & 1) != 0) {
            throw new IllegalArgumentException("Размер кадра должен быть чётным.");
        }

        this.width = width;
        this.height = height;
        this.fps = fps;
        this.yuvFrame = new byte[width * height * 3 / 2];

        codec = MediaCodec.createEncoderByType(MIME);
        int colorFormat = chooseColorFormat(
                codec.getCodecInfo().getCapabilitiesForType(MIME).colorFormats
        );
        semiPlanar = colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar;

        MediaFormat format = MediaFormat.createVideoFormat(MIME, width, height);
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, colorFormat);
        format.setInteger(MediaFormat.KEY_BIT_RATE, bitrate);
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps);
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);

        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        codec.start();

        File parent = output.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("Не удалось создать папку для видео.");
        }
        muxer = new MediaMuxer(
                output.getAbsolutePath(),
                MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4
        );
    }

    public void writeRgbaFrame(byte[] rgba) throws Exception {
        if (finished) throw new IllegalStateException("Encoder уже завершён.");
        int expected = width * height * 4;
        if (rgba == null || rgba.length != expected) {
            throw new IllegalArgumentException(
                    "Неверный RGBA кадр: нужен массив " + expected + " байт."
            );
        }

        rgbaToYuv420(rgba, yuvFrame, width, height, semiPlanar);
        drain(false);

        while (true) {
            int index = codec.dequeueInputBuffer(TIMEOUT_US);
            if (index < 0) {
                drain(false);
                continue;
            }

            ByteBuffer input = codec.getInputBuffer(index);
            if (input == null) {
                throw new IllegalStateException("MediaCodec не вернул input buffer.");
            }
            input.clear();
            if (input.remaining() < yuvFrame.length) {
                throw new IllegalStateException("Input buffer MediaCodec слишком мал.");
            }
            input.put(yuvFrame);

            long ptsUs = frameIndex * 1_000_000L / fps;
            codec.queueInputBuffer(index, 0, yuvFrame.length, ptsUs, 0);
            frameIndex++;
            break;
        }
    }

    public void finish() throws Exception {
        if (finished) return;

        while (true) {
            int index = codec.dequeueInputBuffer(TIMEOUT_US);
            if (index >= 0) {
                long ptsUs = frameIndex * 1_000_000L / fps;
                codec.queueInputBuffer(
                        index,
                        0,
                        0,
                        ptsUs,
                        MediaCodec.BUFFER_FLAG_END_OF_STREAM
                );
                break;
            }
            drain(false);
        }

        drain(true);
        finished = true;
    }

    private void drain(boolean endOfStream) throws Exception {
        int emptyPolls = 0;

        while (true) {
            int index = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US);

            if (index == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (!endOfStream || ++emptyPolls >= 100) return;
                continue;
            }

            if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (muxerStarted) {
                    throw new IllegalStateException("Формат MediaCodec изменился дважды.");
                }
                trackIndex = muxer.addTrack(codec.getOutputFormat());
                muxer.start();
                muxerStarted = true;
                continue;
            }

            if (index < 0) continue;

            ByteBuffer encoded = codec.getOutputBuffer(index);
            if (encoded == null) {
                throw new IllegalStateException("MediaCodec не вернул output buffer.");
            }

            if ((bufferInfo.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                bufferInfo.size = 0;
            }

            if (bufferInfo.size > 0) {
                if (!muxerStarted) {
                    throw new IllegalStateException("MediaMuxer ещё не запущен.");
                }
                encoded.position(bufferInfo.offset);
                encoded.limit(bufferInfo.offset + bufferInfo.size);
                muxer.writeSampleData(trackIndex, encoded, bufferInfo);
            }

            boolean eos = (bufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
            codec.releaseOutputBuffer(index, false);
            if (eos) return;
        }
    }

    private static int chooseColorFormat(int[] formats) {
        boolean flexible = false;
        boolean planar = false;

        for (int format : formats) {
            if (format == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar) {
                return format;
            }
            if (format == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar) {
                planar = true;
            }
            if (format == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible) {
                flexible = true;
            }
        }

        if (planar) {
            return MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar;
        }
        if (flexible) {
            return MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible;
        }
        throw new IllegalStateException("Устройство не предлагает YUV420 для H.264 encoder.");
    }

    private static void rgbaToYuv420(
            byte[] rgba,
            byte[] out,
            int width,
            int height,
            boolean semiPlanar
    ) {
        int frameSize = width * height;
        int yIndex = 0;

        for (int y = 0; y < height; y++) {
            int row = y * width;
            for (int x = 0; x < width; x++) {
                int p = (row + x) * 4;
                int r = rgba[p] & 0xff;
                int g = rgba[p + 1] & 0xff;
                int b = rgba[p + 2] & 0xff;

                int yy = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                out[yIndex++] = (byte) clamp(yy);
            }
        }

        if (semiPlanar) {
            int uv = frameSize;
            for (int y = 0; y < height; y += 2) {
                for (int x = 0; x < width; x += 2) {
                    int[] rgb = averageBlock(rgba, width, height, x, y);
                    int u = ((-38 * rgb[0] - 74 * rgb[1] + 112 * rgb[2] + 128) >> 8) + 128;
                    int v = ((112 * rgb[0] - 94 * rgb[1] - 18 * rgb[2] + 128) >> 8) + 128;
                    out[uv++] = (byte) clamp(u);
                    out[uv++] = (byte) clamp(v);
                }
            }
        } else {
            int uIndex = frameSize;
            int vIndex = frameSize + frameSize / 4;
            for (int y = 0; y < height; y += 2) {
                for (int x = 0; x < width; x += 2) {
                    int[] rgb = averageBlock(rgba, width, height, x, y);
                    int u = ((-38 * rgb[0] - 74 * rgb[1] + 112 * rgb[2] + 128) >> 8) + 128;
                    int v = ((112 * rgb[0] - 94 * rgb[1] - 18 * rgb[2] + 128) >> 8) + 128;
                    out[uIndex++] = (byte) clamp(u);
                    out[vIndex++] = (byte) clamp(v);
                }
            }
        }
    }

    private static int[] averageBlock(
            byte[] rgba,
            int width,
            int height,
            int startX,
            int startY
    ) {
        int r = 0;
        int g = 0;
        int b = 0;
        int count = 0;

        for (int dy = 0; dy < 2; dy++) {
            int y = startY + dy;
            if (y >= height) continue;
            for (int dx = 0; dx < 2; dx++) {
                int x = startX + dx;
                if (x >= width) continue;
                int p = (y * width + x) * 4;
                r += rgba[p] & 0xff;
                g += rgba[p + 1] & 0xff;
                b += rgba[p + 2] & 0xff;
                count++;
            }
        }

        return new int[] { r / count, g / count, b / count };
    }

    private static int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }

    @Override
    public void close() {
        try {
            codec.stop();
        } catch (Exception ignored) {
        }
        try {
            codec.release();
        } catch (Exception ignored) {
        }
        try {
            if (muxerStarted) muxer.stop();
        } catch (Exception ignored) {
        }
        try {
            muxer.release();
        } catch (Exception ignored) {
        }
    }
}
