package com.alex.aivideo;

import android.content.Context;

import java.io.File;

import ai.onnxruntime.OrtEnvironment;

public final class MobileVideoEngine {
    private final Context context;

    public MobileVideoEngine(Context context) {
        this.context = context.getApplicationContext();
    }

    public boolean runtimeReady() {
        try {
            OrtEnvironment.getEnvironment();
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    public File modelDirectory() {
        return new File(context.getFilesDir(), "models/" + ModelPackInstaller.PACK_ID);
    }

    public PackMetadata metadataOrNull() {
        try {
            return ModelPackInstaller.validatePack(modelDirectory());
        } catch (Exception ignored) {
            return null;
        }
    }

    public String modelPackStatus() {
        File dir = modelDirectory();
        if (!dir.exists()) {
            return "MobileI2V model pack ещё не установлен.";
        }

        try {
            PackMetadata metadata = ModelPackInstaller.validatePack(dir);
            return "MobileI2V pack: OK\n" + metadata.summary();
        } catch (Exception e) {
            return "Model pack не прошёл проверку:\n" + e.getMessage();
        }
    }

    public boolean modelPackReady() {
        return metadataOrNull() != null;
    }

    public boolean supportsPrompt() {
        PackMetadata metadata = metadataOrNull();
        return metadata != null && metadata.textConditioning;
    }

    public String deepModelCheck() throws Exception {
        if (!modelPackReady()) {
            throw new IllegalStateException("Model pack ещё не установлен.");
        }
        return new StagedOrtRunner(modelDirectory()).validateModelFiles();
    }

    public File newOutputFile() {
        File dir = new File(context.getExternalFilesDir(null), "generated");
        if (!dir.exists()) {
            //noinspection ResultOfMethodCallIgnored
            dir.mkdirs();
        }
        return new File(dir, "mobilei2v_" + System.currentTimeMillis() + ".mp4");
    }
}
