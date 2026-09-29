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

    public String modelPackStatus() {
        File dir = modelDirectory();
        if (!dir.exists()) {
            return "MobileI2V model pack ещё не установлен.";
        }

        int present = 0;
        for (String name : ModelPackInstaller.REQUIRED_FILES) {
            if (new File(dir, name).isFile()) {
                present++;
            }
        }

        if (present == ModelPackInstaller.REQUIRED_FILES.length) {
            return "MobileI2V pack найден: " + present + "/"
                    + ModelPackInstaller.REQUIRED_FILES.length + " файлов.";
        }
        return "Model pack неполный: " + present + "/"
                + ModelPackInstaller.REQUIRED_FILES.length + " файлов.";
    }

    public boolean modelPackReady() {
        File dir = modelDirectory();
        if (!dir.exists()) return false;
        for (String name : ModelPackInstaller.REQUIRED_FILES) {
            if (!new File(dir, name).isFile()) return false;
        }
        return true;
    }

    /**
     * Loads each ONNX graph separately and closes it immediately.
     * This is intentionally manual, because parsing four large models on app
     * start would create unnecessary memory pressure.
     */
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
