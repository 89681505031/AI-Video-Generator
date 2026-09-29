package com.alex.aivideo;

import android.app.Activity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends Activity {
    private TextView deviceInfo;
    private TextView engineStatus;
    private EditText prompt;
    private Button generateButton;
    private MobileVideoEngine engine;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        deviceInfo = findViewById(R.id.deviceInfo);
        engineStatus = findViewById(R.id.engineStatus);
        prompt = findViewById(R.id.prompt);
        generateButton = findViewById(R.id.generateButton);
        Button checkButton = findViewById(R.id.checkButton);

        DeviceProfile profile = DeviceProfile.detect(this);
        deviceInfo.setText(profile.summary());

        engine = new MobileVideoEngine(this);
        refreshStatus();

        checkButton.setOnClickListener(v -> refreshStatus());
        generateButton.setOnClickListener(v -> {
            if (!engine.modelPackReady()) {
                Toast.makeText(
                        this,
                        "Сначала нужен совместимый мобильный model pack. Сейчас приложение не выдаёт фальшивое AI-видео.",
                        Toast.LENGTH_LONG
                ).show();
                return;
            }

            String text = prompt.getText().toString().trim();
            if (text.isEmpty()) {
                Toast.makeText(this, "Введите описание видео.", Toast.LENGTH_SHORT).show();
                return;
            }

            Toast.makeText(
                    this,
                    "Model pack найден. Следующий этап — подключить реальный inference pipeline и MediaCodec.",
                    Toast.LENGTH_LONG
            ).show();
        });
    }

    private void refreshStatus() {
        boolean ort = engine.runtimeReady();
        String status = (ort ? "ONNX Runtime: OK" : "ONNX Runtime: ошибка")
                + "\n" + engine.modelPackStatus()
                + "\nПапка: " + engine.modelDirectory().getAbsolutePath();

        engineStatus.setText(status);
        generateButton.setEnabled(ort);
    }
}
