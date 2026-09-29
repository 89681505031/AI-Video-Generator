package com.alex.aivideo;

import java.util.Random;

/**
 * Creates the Gaussian video latent used by the official MobileI2V sampler.
 */
public final class LatentInitializer {
    private LatentInitializer() {}

    public static float[] createNoise(
            int width,
            int height,
            long seed
    ) {
        int elements = MobileI2VContract.latentElementCount(width, height);
        float[] latent = new float[elements];
        Random random = new Random(seed);

        for (int i = 0; i < latent.length; i++) {
            latent[i] = (float) random.nextGaussian();
        }
        return latent;
    }

    public static float[] createConditionedNoise(
            int width,
            int height,
            long seed,
            float[] guideImage
    ) {
        float[] latent = createNoise(width, height, seed);
        FlowMatchEulerScheduler.pinGuideImage(
                latent,
                guideImage,
                width,
                height
        );
        return latent;
    }
}
