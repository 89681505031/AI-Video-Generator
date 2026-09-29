package com.alex.aivideo;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;

public class MainActivity extends Activity {
    private static final int REQ_IMAGE = 1001;
    private static final int REQ_MODEL_PACK = 1002;

    private TextView deviceInfo;
    private TextView engineStatus;
    private TextView imageStatus;
    private EditText prompt;
    private Button generateButton;
    private Button downloadModelButton;
    private ModelDownloadController modelDownloadController;
    private MobileVideoEngine engine;
    private Uri selectedImageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        deviceInfo = findViewById(R.id.deviceInfo);
        engineStatus = findViewById(R.id.engineStatus);
        imageStatus = findViewById(R.id.imageStatus);
        prompt = findViewById(R.id.prompt);
        generateButton = findViewById(R.id.generateButton);
        downloadModelButton = findViewById(R.id.downloadModelButton);

        Button checkButton = findViewById(R.id.checkButton);
        Button imageButton = findViewById(R.id.imageButton);
        Button modelPackButton = findViewById(R.id.modelPackButton);

        DeviceProfile profile = DeviceProfile.detect(this);
        deviceInfo.setText(profile.summary());

        engine = new MobileVideoEngine(this);
        modelDownloadController = new ModelDownloadController(
                this,
                new ModelDownloadController.Listener() {
                    @Override
                    public void onStatus(String status) {
                        engineStatus.setText(status);
                    }

                    @Override
                    public void onInstalled() {
                        Toast.makeText(
                                MainActivity.this,
                                "Модель установлена и готова.",
                                Toast.LENGTH_LONG
                        ).show();
                        refreshStatus();
                    }

                    @Override
                    public void onActiveChanged(boolean active) {
                        updateDownloadButton();
                    }
                }
        );
        refreshStatus();
        modelDownloadController.resume();

        checkButton.setOnClickListener(v -> runDeepCheck());
        imageButton.setOnClickListener(v -> openImagePicker());
        modelPackButton.setOnClickListener(v -> openModelPackPicker());
        downloadModelButton.setOnClickListener(v -> modelDownloadController.start());
        generateButton.setOnClickListener(v -> generateVideo());
    }

    private void generateVideo() {
        PackMetadata metadata = engine.metadataOrNull();
        if (metadata == null) {
            Toast.makeText(
                    this,
                    "Сначала установите проверенный MobileI2V model pack.",
                    Toast.LENGTH_LONG
            ).show();
            return;
        }

        if (selectedImageUri == null) {
            Toast.makeText(
                    this,
                    "Сначала выберите исходное изображение.",
                    Toast.LENGTH_SHORT
            ).show();
            return;
        }

        generateButton.setEnabled(false);
        engineStatus.setText("0% • Запуск генерации");

        new Thread(() -> {
            try {
                File output = engine.generate(
                        selectedImageUri,
                        (percent, stage) -> runOnUiThread(() ->
                                engineStatus.setText(
                                        percent + "% • " + stage
                                )
                        )
                );

                runOnUiThread(() -> {
                    engineStatus.setText(
                            "100% • Видео готово\n" + output.getAbsolutePath()
                    );
                    generateButton.setEnabled(true);
                    Toast.makeText(
                            this,
                            "Видео создано на телефоне.",
                            Toast.LENGTH_LONG
                    ).show();
                });
            } catch (Throwable e) {
                runOnUiThread(() -> {
                    engineStatus.setText(
                            "Ошибка генерации:\n"
                                    + (e.getMessage() == null
                                    ? e.getClass().getSimpleName()
                                    : e.getMessage())
                    );
                    generateButton.setEnabled(true);
                });
            }
        }, "mobilei2v-generation").start();
    }

    private void openImagePicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        intent.addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(intent, REQ_IMAGE);
    }

    private void openModelPackPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivityForResult(intent, REQ_MODEL_PACK);
    }

    @Override
    protected void onActivityResult(
            int requestCode,
            int resultCode,
            Intent data
    ) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode != RESULT_OK
                || data == null
                || data.getData() == null) {
            return;
        }

        Uri uri = data.getData();

        if (requestCode == REQ_IMAGE) {
            selectedImageUri = uri;
            try {
                int flags = data.getFlags()
                        & Intent.FLAG_GRANT_READ_URI_PERMISSION;
                if (flags != 0) {
                    getContentResolver().takePersistableUriPermission(
                            uri,
                            flags
                    );
                }
            } catch (SecurityException ignored) {
            }

            imageStatus.setText("Изображение выбрано:\n" + uri);
            return;
        }

        if (requestCode == REQ_MODEL_PACK) {
            installModelPack(uri);
        }
    }

    private void installModelPack(Uri uri) {
        engineStatus.setText(
                "Проверяю ZIP, runtime contract и SHA-256…"
        );
        generateButton.setEnabled(false);

        new Thread(() -> {
            try {
                ModelPackInstaller.install(this, uri);
                runOnUiThread(() -> {
                    Toast.makeText(
                            this,
                            "MobileI2V model pack установлен.",
                            Toast.LENGTH_LONG
                    ).show();
                    refreshStatus();
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    engineStatus.setText(
                            "Model pack отклонён:\n" + e.getMessage()
                    );
                    generateButton.setEnabled(engine.runtimeReady());
                });
            }
        }, "model-pack-installer").start();
    }

    private void runDeepCheck() {
        if (!engine.runtimeReady()) {
            engineStatus.setText("ONNX Runtime: ошибка");
            return;
        }
        if (!engine.modelPackReady()) {
            refreshStatus();
            return;
        }

        engineStatus.setText(
                "Поочерёдно проверяю ONNX-модели…"
        );
        generateButton.setEnabled(false);

        new Thread(() -> {
            try {
                String report = engine.deepModelCheck();
                runOnUiThread(() -> {
                    engineStatus.setText(
                            "ONNX Runtime: OK\n"
                                    + engine.modelPackStatus()
                                    + "\n\nГлубокая проверка:\n"
                                    + report
                    );
                    generateButton.setEnabled(true);
                });
            } catch (Throwable e) {
                runOnUiThread(() -> {
                    engineStatus.setText(
                            "Ошибка проверки ONNX:\n" + e.getMessage()
                    );
                    generateButton.setEnabled(true);
                });
            }
        }, "ort-deep-check").start();
    }

    private void updateDownloadButton() {
        boolean ready = engine != null && engine.modelPackReady();
        boolean active = modelDownloadController != null
                && modelDownloadController.isActive();

        downloadModelButton.setEnabled(!ready && !active);
        downloadModelButton.setText(
                ready
                        ? "Модель установлена ✓"
                        : (active
                        ? "Модель скачивается…"
                        : "Скачать модель 1.7 ГБ")
        );
    }

    private void refreshStatus() {
        boolean ort = engine.runtimeReady();

        String status = (ort ? "ONNX Runtime: OK" : "ONNX Runtime: ошибка")
                + "\n" + engine.modelPackStatus()
                + "\nПапка: "
                + engine.modelDirectory().getAbsolutePath();

        engineStatus.setText(status);

        prompt.setEnabled(false);
        prompt.setText("");
        prompt.setHint(
                "Текущий официальный MobileI2V: prompt не используется"
        );

        generateButton.setEnabled(
                ort && engine.modelPackReady()
        );
        updateDownloadButton();
    }

    @Override
    protected void onDestroy() {
        if (modelDownloadController != null) {
            modelDownloadController.close();
        }
        super.onDestroy();
    }
}
