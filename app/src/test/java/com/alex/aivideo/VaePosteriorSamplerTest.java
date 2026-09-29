package com.alex.aivideo;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class VaePosteriorSamplerTest {
    @Test
    public void posteriorSamplingIsDeterministicForSeed() {
        int width = 32;
        int height = 32;
        int guideElements = MobileI2VContract.guideElementCount(width, height);
        float[] moments = new float[guideElements * 2];

        float[] a = VaePosteriorSampler.sampleGuideLatent(
                moments,
                width,
                height,
                1234L
        );
        float[] b = VaePosteriorSampler.sampleGuideLatent(
                moments,
                width,
                height,
                1234L
        );

        assertArrayEquals(a, b, 0.0f);
    }

    @Test
    public void posteriorSamplingAppliesMeanAndScale() {
        int width = 32;
        int height = 32;
        int guideElements = MobileI2VContract.guideElementCount(width, height);
        float[] moments = new float[guideElements * 2];

        for (int i = 0; i < guideElements; i++) {
            moments[i] = 2.0f;
            moments[guideElements + i] = -30.0f;
        }

        float[] sampled = VaePosteriorSampler.sampleGuideLatent(
                moments,
                width,
                height,
                42L
        );

        assertEquals(
                2.0f * MobileI2VContract.VAE_SCALE_FACTOR,
                sampled[0],
                1e-5f
        );
    }

    @Test
    public void conditionedNoisePinsFirstTemporalSlice() {
        int width = 32;
        int height = 32;
        int guideElements = MobileI2VContract.guideElementCount(width, height);
        float[] guide = new float[guideElements];
        for (int i = 0; i < guide.length; i++) {
            guide[i] = i + 0.25f;
        }

        float[] latent = LatentInitializer.createConditionedNoise(
                width,
                height,
                7L,
                guide
        );

        int spatial = MobileI2VContract.spatialPositions(width, height);
        int time = MobileI2VContract.LATENT_TIME;

        for (int c = 0; c < MobileI2VContract.LATENT_CHANNELS; c++) {
            int latentOffset = c * time * spatial;
            int guideOffset = c * spatial;
            for (int i = 0; i < spatial; i++) {
                assertEquals(
                        guide[guideOffset + i],
                        latent[latentOffset + i],
                        0.0f
                );
            }

            if (time > 1) {
                assertNotEquals(
                        guide[guideOffset],
                        latent[latentOffset + spatial],
                        0.0f
                );
            }
        }
    }
}
