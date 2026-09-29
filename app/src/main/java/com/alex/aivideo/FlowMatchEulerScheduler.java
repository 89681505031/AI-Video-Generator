package com.alex.aivideo;

/**
 * Deterministic Java port of Diffusers 0.35.2
 * FlowMatchEulerDiscreteScheduler used by MobileI2V.
 */
public final class FlowMatchEulerScheduler {
    private static final int TRAIN_TIMESTEPS = 1000;
    private static final double SHIFT = MobileI2VContract.FLOW_SHIFT;

    private final float[] sigmas;
    private final float[] timesteps;

    public FlowMatchEulerScheduler(int inferenceSteps) {
        if (inferenceSteps < 1) {
            throw new IllegalArgumentException("inferenceSteps должен быть >= 1.");
        }

        sigmas = new float[inferenceSteps + 1];
        timesteps = new float[inferenceSteps];

        double sigmaMax = shiftSigma(1.0);
        double sigmaMin = shiftSigma(1.0 / TRAIN_TIMESTEPS);

        for (int i = 0; i < inferenceSteps; i++) {
            double fraction = inferenceSteps == 1
                    ? 0.0
                    : (double) i / (double) (inferenceSteps - 1);
            double initialT = lerp(
                    sigmaMax * TRAIN_TIMESTEPS,
                    sigmaMin * TRAIN_TIMESTEPS,
                    fraction
            );
            double rawSigma = initialT / TRAIN_TIMESTEPS;
            double sigma = shiftSigma(rawSigma);
            sigmas[i] = (float) sigma;
            timesteps[i] = (float) (sigma * TRAIN_TIMESTEPS);
        }
        sigmas[inferenceSteps] = 0.0f;
    }

    public int steps() {
        return timesteps.length;
    }

    public float timestep(int index) {
        checkStep(index);
        return timesteps[index];
    }

    public float sigma(int index) {
        if (index < 0 || index >= sigmas.length) {
            throw new IndexOutOfBoundsException("sigma index=" + index);
        }
        return sigmas[index];
    }

    public void stepInPlace(float[] sample, float[] modelOutput, int stepIndex) {
        checkStep(stepIndex);
        if (sample == null || modelOutput == null || sample.length != modelOutput.length) {
            throw new IllegalArgumentException(
                    "sample/modelOutput должны иметь одинаковую длину."
            );
        }
        float dt = sigmas[stepIndex + 1] - sigmas[stepIndex];
        for (int i = 0; i < sample.length; i++) {
            sample[i] += dt * modelOutput[i];
        }
    }

    public static void classifierFreeGuidance(
            float[] unconditioned,
            float[] conditioned,
            float cfgScale,
            float[] output
    ) {
        if (unconditioned == null || conditioned == null || output == null
                || unconditioned.length != conditioned.length
                || output.length != unconditioned.length) {
            throw new IllegalArgumentException("CFG tensors должны иметь одинаковую длину.");
        }
        for (int i = 0; i < output.length; i++) {
            output[i] = unconditioned[i]
                    + cfgScale * (conditioned[i] - unconditioned[i]);
        }
    }

    public static void pinGuideImage(
            float[] latent,
            float[] guideImage,
            int width,
            int height
    ) {
        int expectedLatent = MobileI2VContract.latentElementCount(width, height);
        int expectedGuide = MobileI2VContract.guideElementCount(width, height);
        if (latent == null || latent.length != expectedLatent) {
            throw new IllegalArgumentException(
                    "Неверный размер latent: ожидалось " + expectedLatent + "."
            );
        }
        if (guideImage == null || guideImage.length != expectedGuide) {
            throw new IllegalArgumentException(
                    "Неверный размер guideImage: ожидалось " + expectedGuide + "."
            );
        }

        int spatial = MobileI2VContract.spatialPositions(width, height);
        int time = MobileI2VContract.LATENT_TIME;
        for (int c = 0; c < MobileI2VContract.LATENT_CHANNELS; c++) {
            int latentOffset = c * time * spatial;
            int guideOffset = c * spatial;
            System.arraycopy(
                    guideImage, guideOffset, latent, latentOffset, spatial
            );
        }
    }

    private void checkStep(int index) {
        if (index < 0 || index >= timesteps.length) {
            throw new IndexOutOfBoundsException("step index=" + index);
        }
    }

    private static double shiftSigma(double sigma) {
        return SHIFT * sigma / (1.0 + (SHIFT - 1.0) * sigma);
    }

    private static double lerp(double start, double end, double t) {
        return start + (end - start) * t;
    }
}
