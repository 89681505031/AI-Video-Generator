package com.alex.aivideo;

import java.io.File;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * Opens only one ONNX model at a time.
 *
 * This is deliberate for phones: the video pipeline contains several large
 * models and keeping all sessions alive at once can cause avoidable RAM peaks.
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
     * Verifies that every neural-network file can at least be parsed by ORT.
     * Sessions are opened and immediately closed one-by-one.
     */
    public String validateModelFiles() throws Exception {
        String[] models = {
                "vae_encoder.onnx",
                "qwen2_encoder.onnx",
                "mobilei2v_unet.onnx",
                "turbo_vaed.onnx"
        };

        StringBuilder result = new StringBuilder();
        for (String name : models) {
            withSession(name, session -> null);
            if (result.length() > 0) result.append('\n');
            result.append("OK • ").append(name);
        }
        return result.toString();
    }
}
