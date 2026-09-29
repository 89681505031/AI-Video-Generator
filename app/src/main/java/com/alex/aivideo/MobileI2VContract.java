package com.alex.aivideo;

/** Geometry/constants verified against the official HUST MobileI2V source. */
public final class MobileI2VContract {
    public static final int OUTPUT_FRAMES = 17;
    public static final int OUTPUT_FPS = 17;
    public static final int LATENT_CHANNELS = 128;
    public static final int SPATIAL_DOWNSAMPLE = 32;
    public static final int TEMPORAL_DOWNSAMPLE = 8;
    public static final int LATENT_TIME =
            OUTPUT_FRAMES / TEMPORAL_DOWNSAMPLE + 1;
    public static final int TEXT_MAX_TOKENS = 300;
    public static final int TEXT_CHANNELS = 896;
    public static final float VAE_SCALE_FACTOR = 0.41407f;
    public static final float FLOW_SHIFT = 1.0f;
    public static final float DEFAULT_FLOW_SCORE = 2.0f;

    private MobileI2VContract() {}

    public static int latentWidth(int width) {
        return ceilDiv(width, SPATIAL_DOWNSAMPLE);
    }

    public static int latentHeight(int height) {
        return ceilDiv(height, SPATIAL_DOWNSAMPLE);
    }

    public static int spatialPositions(int width, int height) {
        return latentWidth(width) * latentHeight(height);
    }

    public static int sequencePositions(int width, int height) {
        return LATENT_TIME * spatialPositions(width, height);
    }

    public static int latentElementCount(int width, int height) {
        return LATENT_CHANNELS * sequencePositions(width, height);
    }

    public static int guideElementCount(int width, int height) {
        return LATENT_CHANNELS * spatialPositions(width, height);
    }

    public static float[] createConditionMask(int width, int height) {
        float[] mask = new float[sequencePositions(width, height)];
        int conditioned = spatialPositions(width, height);
        for (int i = 0; i < conditioned; i++) {
            mask[i] = 1.0f;
        }
        return mask;
    }

    public static int ceilDiv(int value, int divisor) {
        if (value <= 0 || divisor <= 0) {
            throw new IllegalArgumentException("value/divisor должны быть > 0.");
        }
        return (value + divisor - 1) / divisor;
    }
}
