package com.alex.aivideo;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Map;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * Real ONNX Runtime core for official MobileI2V.
 */
public final class MobileI2VOrtCore {
    private static final Set<String> VAE_INPUTS = Set.of("image");
    private static final Set<String> VAE_OUTPUTS = Set.of("posterior_moments");

    private static final Set<String> DENOISER_INPUTS = Set.of(
            "x",
            "timestep",
            "cond_mask",
            "flow_score"
    );
    private static final Set<String> DENOISER_OUTPUTS = Set.of("noise");

    private static final Set<String> DECODER_INPUTS = Set.of("latent");
    private static final Set<String> DECODER_OUTPUTS = Set.of("video");

    private final OrtEnvironment environment;
    private final StagedOrtRunner runner;

    public MobileI2VOrtCore(File modelDirectory) {
        environment = OrtEnvironment.getEnvironment();
        runner = new StagedOrtRunner(modelDirectory);
    }

    public float[] encodeGuideImage(
            float[] imageNcthw,
            PackMetadata metadata,
            long posteriorSeed
    ) throws Exception {
        int expectedImageElements = 3 * metadata.width * metadata.height;
        if (imageNcthw == null || imageNcthw.length != expectedImageElements) {
            throw new IllegalArgumentException(
                    "Неверный image tensor: ожидалось "
                            + expectedImageElements + " float-элементов."
            );
        }

        float[] moments = runner.withSession("vae_encoder.onnx", session -> {
            requireContract(session, VAE_INPUTS, VAE_OUTPUTS, "VAE encoder");

            long[] imageShape = {
                    1,
                    3,
                    1,
                    metadata.height,
                    metadata.width
            };

            try (OnnxTensor image = OrtTensorIO.floatTensor(
                    environment,
                    imageNcthw,
                    imageShape
            )) {
                try (OrtSession.Result result =
                             session.run(Map.of("image", image))) {
                    OnnxValue output = result.get("posterior_moments")
                            .orElseThrow(() -> new IllegalStateException(
                                    "VAE encoder не вернул posterior_moments."
                            ));

                    return OrtTensorIO.copyFloatOutput(
                            output,
                            MobileI2VContract.guideElementCount(
                                    metadata.width,
                                    metadata.height
                            ) * 2
                    );
                }
            }
        });

        return VaePosteriorSampler.sampleGuideLatent(
                moments,
                metadata.width,
                metadata.height,
                posteriorSeed
        );
    }

    public float[] denoise(
            float[] initialLatent,
            float[] guideImage,
            PackMetadata metadata,
            Progress progress
    ) throws Exception {
        int latentElements = MobileI2VContract.latentElementCount(
                metadata.width,
                metadata.height
        );
        int guideElements = MobileI2VContract.guideElementCount(
                metadata.width,
                metadata.height
        );

        if (initialLatent == null || initialLatent.length != latentElements) {
            throw new IllegalArgumentException("Неверный initial latent.");
        }
        if (guideImage == null || guideImage.length != guideElements) {
            throw new IllegalArgumentException("Неверный guide latent.");
        }

        float[] latent = initialLatent.clone();
        FlowMatchEulerScheduler scheduler =
                new FlowMatchEulerScheduler(metadata.samplingSteps);

        return runner.withSession("mobilei2v_transformer.onnx", session -> {
            requireContract(
                    session,
                    DENOISER_INPUTS,
                    DENOISER_OUTPUTS,
                    "MobileDiT"
            );

            float[] condMask = MobileI2VContract.createConditionMask(
                    metadata.width,
                    metadata.height
            );
            int latentHeight = metadata.latentHeight();
            int latentWidth = metadata.latentWidth();

            try (OnnxTensor conditionMask = OrtTensorIO.floatTensor(
                         environment,
                         condMask,
                         new long[] {1, metadata.sequencePositions()}
                 );
                 OnnxTensor flowScore = OrtTensorIO.floatTensor(
                         environment,
                         new float[] {MobileI2VContract.DEFAULT_FLOW_SCORE},
                         new long[] {1}
                 )) {

                for (int step = 0; step < scheduler.steps(); step++) {
                    if (progress != null) {
                        progress.onStep(step, scheduler.steps());
                    }

                    try (OnnxTensor x = OrtTensorIO.floatTensor(
                                 environment,
                                 latent,
                                 new long[] {
                                         1,
                                         MobileI2VContract.LATENT_CHANNELS,
                                         MobileI2VContract.LATENT_TIME,
                                         latentHeight,
                                         latentWidth
                                 }
                         );
                         OnnxTensor timestep = OrtTensorIO.floatTensor(
                                 environment,
                                 new float[] {scheduler.timestep(step)},
                                 new long[] {1}
                         )) {

                        Map<String, OnnxTensor> inputs = Map.of(
                                "x", x,
                                "timestep", timestep,
                                "cond_mask", conditionMask,
                                "flow_score", flowScore
                        );

                        try (OrtSession.Result result = session.run(inputs)) {
                            OnnxValue output = result.get("noise")
                                    .orElseThrow(() ->
                                            new IllegalStateException(
                                                    "MobileDiT не вернул noise."
                                            )
                                    );

                            float[] noise = OrtTensorIO.copyFloatOutput(
                                    output,
                                    latentElements
                            );

                            scheduler.stepInPlace(latent, noise, step);
                            FlowMatchEulerScheduler.pinGuideImage(
                                    latent,
                                    guideImage,
                                    metadata.width,
                                    metadata.height
                            );
                        }
                    }
                }
            }

            return latent;
        });
    }

    public void decodeToMp4(
            float[] latent,
            PackMetadata metadata,
            File outputFile,
            FrameProgress progress
    ) throws Exception {
        int expectedLatent = MobileI2VContract.latentElementCount(
                metadata.width,
                metadata.height
        );
        if (latent == null || latent.length != expectedLatent) {
            throw new IllegalArgumentException("Неверный final latent.");
        }

        runner.withSession("video_decoder.onnx", session -> {
            requireContract(
                    session,
                    DECODER_INPUTS,
                    DECODER_OUTPUTS,
                    "Video decoder"
            );

            long[] latentShape = {
                    1,
                    MobileI2VContract.LATENT_CHANNELS,
                    MobileI2VContract.LATENT_TIME,
                    metadata.latentHeight(),
                    metadata.latentWidth()
            };

            try (OnnxTensor input = OrtTensorIO.floatTensor(
                    environment,
                    latent,
                    latentShape
            )) {
                try (OrtSession.Result result =
                             session.run(Map.of("latent", input))) {
                    OnnxValue output = result.get("video")
                            .orElseThrow(() -> new IllegalStateException(
                                    "Video decoder не вернул video."
                            ));

                    if (!(output instanceof OnnxTensor)) {
                        throw new IllegalStateException(
                                "Decoder output не является tensor."
                        );
                    }

                    FloatBuffer video =
                            ((OnnxTensor) output).getFloatBuffer();
                    if (video == null) {
                        throw new IllegalStateException(
                                "Decoder output должен быть FP32 для pack v4."
                        );
                    }

                    int expectedVideoElements =
                            3
                                    * MobileI2VContract.OUTPUT_FRAMES
                                    * metadata.width
                                    * metadata.height;

                    if (video.capacity() < expectedVideoElements) {
                        throw new IllegalStateException(
                                "Decoder output содержит "
                                        + video.capacity()
                                        + " float-элементов, ожидалось >= "
                                        + expectedVideoElements
                        );
                    }

                    int bitrate = Math.max(
                            2_000_000,
                            Math.min(
                                    12_000_000,
                                    metadata.width
                                            * metadata.height
                                            * MobileI2VContract.OUTPUT_FPS
                            )
                    );

                    try (H264Mp4Encoder encoder = new H264Mp4Encoder(
                            outputFile,
                            metadata.width,
                            metadata.height,
                            MobileI2VContract.OUTPUT_FPS,
                            bitrate
                    )) {
                        for (int frame = 0;
                             frame < MobileI2VContract.OUTPUT_FRAMES;
                             frame++) {
                            byte[] rgba = VideoTensorFrames.readRgbaFrame(
                                    video,
                                    frame,
                                    MobileI2VContract.OUTPUT_FRAMES,
                                    metadata.width,
                                    metadata.height
                            );
                            encoder.writeRgbaFrame(rgba);

                            if (progress != null) {
                                progress.onFrame(
                                        frame + 1,
                                        MobileI2VContract.OUTPUT_FRAMES
                                );
                            }
                        }
                        encoder.finish();
                    }
                }
            }

            return null;
        });
    }

    public interface Progress {
        void onStep(int stepIndex, int totalSteps);
    }

    public interface FrameProgress {
        void onFrame(int frameNumber, int totalFrames);
    }

    private static void requireContract(
            OrtSession session,
            Set<String> expectedInputs,
            Set<String> expectedOutputs,
            String modelName
    ) throws Exception {
        Set<String> actualInputs = session.getInputNames();
        Set<String> actualOutputs = session.getOutputNames();

        if (!actualInputs.equals(expectedInputs)) {
            throw new IllegalArgumentException(
                    modelName + " inputs=" + actualInputs
                            + ", ожидалось " + expectedInputs
            );
        }
        if (!actualOutputs.equals(expectedOutputs)) {
            throw new IllegalArgumentException(
                    modelName + " outputs=" + actualOutputs
                            + ", ожидалось " + expectedOutputs
            );
        }
    }
}
