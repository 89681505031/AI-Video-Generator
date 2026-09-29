package com.alex.aivideo;

import org.junit.Test;

import java.nio.FloatBuffer;

import static org.junit.Assert.assertEquals;

public class VideoTensorFramesTest {
    @Test
    public void normalizedConversionClampsCorrectly() {
        assertEquals(0, VideoTensorFrames.normalizedToByte(-2.0f) & 0xff);
        assertEquals(0, VideoTensorFrames.normalizedToByte(-1.0f) & 0xff);
        assertEquals(128, VideoTensorFrames.normalizedToByte(0.0f) & 0xff);
        assertEquals(255, VideoTensorFrames.normalizedToByte(1.0f) & 0xff);
        assertEquals(255, VideoTensorFrames.normalizedToByte(2.0f) & 0xff);
    }

    @Test
    public void readsNcthwFrameIntoRgba() {
        int frames = 2;
        int width = 2;
        int height = 1;
        int spatial = width * height;

        float[] data = new float[3 * frames * spatial];

        // R frame 1
        data[2] = -1.0f;
        data[3] = 1.0f;
        // G frame 1
        data[4 + 2] = 0.0f;
        data[4 + 3] = 0.0f;
        // B frame 1
        data[8 + 2] = 1.0f;
        data[8 + 3] = -1.0f;

        byte[] rgba = VideoTensorFrames.readRgbaFrame(
                FloatBuffer.wrap(data),
                1,
                frames,
                width,
                height
        );

        assertEquals(0, rgba[0] & 0xff);
        assertEquals(128, rgba[1] & 0xff);
        assertEquals(255, rgba[2] & 0xff);
        assertEquals(255, rgba[3] & 0xff);

        assertEquals(255, rgba[4] & 0xff);
        assertEquals(128, rgba[5] & 0xff);
        assertEquals(0, rgba[6] & 0xff);
        assertEquals(255, rgba[7] & 0xff);
    }
}
