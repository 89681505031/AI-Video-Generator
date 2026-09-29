package com.alex.aivideo;

import java.util.Random;

/**
 * Reproduces Diffusers DiagonalGaussianDistribution.sample() for the LTX
 * video-VAE guide image, then applies MobileI2V's VAE scale factor.
 */
public final class VaePosteriorSampler {
    private VaePosteriorSampler() {}

    public static float[] sampleGuideLatent(
            float[] posteriorMoments,
            int width,
            int height,
            long seed
    ) {
        int guideElements = MobileI2VContract.guideElementCount(width, height);
        int expectedMoments = guideElements * 2;

        if (posteriorMoments == null || posteriorMoments.length != expectedMoments) {
            throw new IllegalArgumentException(
                    "Неверный размер posterior moments: ожидалось "
                            + expectedMoments + "."
            );
        }

        float[] latent = new float[guideElements];
        Random random = new Random(seed);

        for (int i = 0; i < guideElements; i++) {
            float mean = posteriorMoments[i];
            float logvar = posteriorMoments[guideElements + i];
            logvar = Math.max(-30.0f, Math.min(20.0f, logvar));

            double std = Math.exp(0.5 * logvar);
            double sample = mean + std * random.nextGaussian();
            latent[i] = (float) (sample * MobileI2VContract.VAE_SCALE_FACTOR);
        }

        return latent;
    }
}
