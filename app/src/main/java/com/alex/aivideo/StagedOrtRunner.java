package com.alex.aivideo;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * Opens only one ONNX model at a time to keep phone RAM peaks down.
 */
public final class StagedOrtRunner {
    public interface SessionTask<T> {
        T run(OrtSession session) throws Exception;
    }

    private final OrtEnvironment environment;
    private final File modelDirectory;

    public StagedOrtRunner(File modelDirectory) {
        this.environment = OrtEnvironment.getEnvironment();
        this.modelDirectory = modelDirectory;
    }

    public <T> T withSession(String fileName, SessionTask<T> task) throws Exception {
        File model = new File(modelDirectory, fileName);
        if (!model.isFile()) {
            throw new IllegalArgumentException("Нет модели: " + fileName);
        }

        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions();
             OrtSession session = environment.createSession(model.getAbsolutePath(), options)) {
            return task.run(session);
        }
    }

    /**
     * Parses each ONNX graph separately and reports its real input/output names.
     */
    public String validateModelFiles() throws Exception {
        PackMetadata metadata = ModelPackInstaller.validatePack(modelDirectory);

        List<String> models = new ArrayList<>();
        models.add("vae_encoder.onnx");
        if (metadata.textConditioning) {
            models.add("qwen2_encoder.onnx");
        }
        models.add("mobilei2v_transformer.onnx");
        models.add("video_decoder.onnx");

        StringBuilder result = new StringBuilder();
        for (String name : models) {
            String io = withSession(name, session ->
                    "inputs=" + session.getInputNames()
                            + " • outputs=" + session.getOutputNames()
            );

            if (result.length() > 0) result.append('\n');
            result.append("OK • ").append(name).append("\n   ").append(io);
        }
        return result.toString();
    }
}
