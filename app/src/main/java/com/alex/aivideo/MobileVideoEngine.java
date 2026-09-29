package com.alex.aivideo;

import android.content.Context;

import java.io.File;

import ai.onnxruntime.OrtEnvironment;

public final class MobileVideoEngine {
    private static final String PACK_DIR = "mobile_video_v1";
    private static final String[] REQUIRED_FILES = {
            "text_encoder.onnx",
            "video_model.onnx",
            "vae_decoder.onnx",
            "tokenizer.json",
            "manifest.json"
    };

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
        return new File(context.getFilesDir(), "models/" + PACK_DIR);
    }

    public String modelPackStatus() {
        File dir = modelDirectory();
        if (!dir.exists()) {
            return "Мобильная модель ещё не установлена.";
        }

        int present = 0;
        for (String name : REQUIRED_FILES) {
            if (new File(dir, name).isFile()) {
                present++;
            }
        }

        if (present == REQUIRED_FILES.length) {
            return "Model pack найден: " + present + "/" + REQUIRED_FILES.length + " файлов.";
        }
        return "Model pack неполный: " + present + "/" + REQUIRED_FILES.length + " файлов.";
    }

    public boolean modelPackReady() {
        File dir = modelDirectory();
        if (!dir.exists()) return false;
        for (String name : REQUIRED_FILES) {
            if (!new File(dir, name).isFile()) return false;
        }
        return true;
    }
}
