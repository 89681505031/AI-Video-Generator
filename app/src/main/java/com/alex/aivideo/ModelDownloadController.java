package com.alex.aivideo;

import android.app.Activity;
import android.app.DownloadManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;

import java.io.File;
import java.util.Locale;

public final class ModelDownloadController {
    public interface Listener {
        void onStatus(String status);
        void onInstalled();
        void onActiveChanged(boolean active);
    }

    private static final String PREFS = "mobile_model_download";
    private static final String PREF_ID = "download_id";

    private final Activity activity;
    private final Listener listener;
    private final DownloadManager manager;
    private final Handler handler = new Handler(Looper.getMainLooper());

    private long activeId = -1L;
    private boolean registered;

    private final Runnable poll = new Runnable() {
        @Override
        public void run() {
            if (activeId < 0) return;
            if (updateProgress(activeId)) {
                handler.postDelayed(this, 1000L);
            } else {
                checkCompleted(activeId);
            }
        }
    };

    private final BroadcastReceiver receiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (!DownloadManager.ACTION_DOWNLOAD_COMPLETE.equals(
                    intent.getAction()
            )) {
                return;
            }

            long id = intent.getLongExtra(
                    DownloadManager.EXTRA_DOWNLOAD_ID,
                    -1L
            );
            if (id == activeId) {
                finishDownload(id);
            }
        }
    };

    public ModelDownloadController(Activity activity, Listener listener) {
        this.activity = activity;
        this.listener = listener;
        this.manager = (DownloadManager) activity.getSystemService(
                Context.DOWNLOAD_SERVICE
        );

        IntentFilter filter = new IntentFilter(
                DownloadManager.ACTION_DOWNLOAD_COMPLETE
        );
        if (Build.VERSION.SDK_INT >= 33) {
            activity.registerReceiver(
                    receiver,
                    filter,
                    Context.RECEIVER_NOT_EXPORTED
            );
        } else {
            activity.registerReceiver(receiver, filter);
        }
        registered = true;

        activeId = activity.getSharedPreferences(
                PREFS,
                Context.MODE_PRIVATE
        ).getLong(PREF_ID, -1L);
    }

    public boolean isActive() {
        return activeId >= 0;
    }

    public void resume() {
        if (activeId < 0) {
            notifyActive();
            return;
        }

        if (updateProgress(activeId)) {
            handler.post(poll);
        } else {
            checkCompleted(activeId);
        }
    }

    public void start() {
        if (activeId >= 0) return;

        try {
            ModelRelease.requireEnoughSpace(activity);

            File destination = ModelRelease.downloadFile(activity);
            if (destination.exists() && !destination.delete()) {
                throw new IllegalStateException(
                        "Не удалось удалить старый ZIP модели."
                );
            }

            DownloadManager.Request request = new DownloadManager.Request(
                    Uri.parse(ModelRelease.DOWNLOAD_URL)
            )
                    .setTitle("MobileI2V 512 model")
                    .setDescription(
                            "Модель для локальной генерации видео • 1.7 ГБ"
                    )
                    .setNotificationVisibility(
                            DownloadManager.Request
                                    .VISIBILITY_VISIBLE_NOTIFY_COMPLETED
                    )
                    .setAllowedOverMetered(true)
                    .setAllowedOverRoaming(false)
                    .setDestinationInExternalFilesDir(
                            activity,
                            Environment.DIRECTORY_DOWNLOADS,
                            ModelRelease.FILE_NAME
                    );

            activeId = manager.enqueue(request);
            activity.getSharedPreferences(
                    PREFS,
                    Context.MODE_PRIVATE
            ).edit().putLong(PREF_ID, activeId).apply();

            notifyActive();
            listener.onStatus("0% • Скачивание модели");
            handler.removeCallbacks(poll);
            handler.post(poll);
        } catch (Throwable error) {
            listener.onStatus("Ошибка скачивания модели:\n" + readable(error));
            clearState();
        }
    }

    private boolean updateProgress(long id) {
        DownloadManager.Query query = new DownloadManager.Query()
                .setFilterById(id);

        try (Cursor cursor = manager.query(query)) {
            if (cursor == null || !cursor.moveToFirst()) {
                return false;
            }

            int status = cursor.getInt(
                    cursor.getColumnIndexOrThrow(
                            DownloadManager.COLUMN_STATUS
                    )
            );

            if (status == DownloadManager.STATUS_SUCCESSFUL
                    || status == DownloadManager.STATUS_FAILED) {
                return false;
            }

            long done = cursor.getLong(
                    cursor.getColumnIndexOrThrow(
                            DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR
                    )
            );
            long total = cursor.getLong(
                    cursor.getColumnIndexOrThrow(
                            DownloadManager.COLUMN_TOTAL_SIZE_BYTES
                    )
            );

            if (total > 0) {
                int percent = (int) Math.min(99L, done * 100L / total);
                listener.onStatus(
                        percent + "% • Скачивание модели • "
                                + humanSize(done)
                                + " / "
                                + humanSize(total)
                );
            } else {
                listener.onStatus(
                        "Скачивание модели • " + humanSize(done)
                );
            }
            return true;
        }
    }

    private void checkCompleted(long id) {
        if (id < 0) return;

        DownloadManager.Query query = new DownloadManager.Query()
                .setFilterById(id);

        try (Cursor cursor = manager.query(query)) {
            if (cursor == null || !cursor.moveToFirst()) {
                clearState();
                return;
            }

            int status = cursor.getInt(
                    cursor.getColumnIndexOrThrow(
                            DownloadManager.COLUMN_STATUS
                    )
            );

            if (status == DownloadManager.STATUS_SUCCESSFUL) {
                finishDownload(id);
            } else if (status == DownloadManager.STATUS_FAILED) {
                int reason = cursor.getInt(
                        cursor.getColumnIndexOrThrow(
                                DownloadManager.COLUMN_REASON
                        )
                );
                listener.onStatus(
                        "Ошибка скачивания модели • код " + reason
                );
                clearState();
            }
        }
    }

    private void finishDownload(long id) {
        handler.removeCallbacks(poll);
        listener.onStatus("Проверяю SHA-256 скачанной модели…");

        new Thread(() -> {
            try {
                File zip = ModelRelease.downloadFile(activity);
                ModelRelease.verifyDownloadedZip(zip);

                Uri uri = manager.getUriForDownloadedFile(id);
                if (uri == null) {
                    uri = Uri.fromFile(zip);
                }

                postStatus("SHA-256: OK • Устанавливаю model pack…");
                ModelPackInstaller.installVerifiedCurrent(activity, uri);

                manager.remove(id);
                clearStateOnMain();

                activity.runOnUiThread(listener::onInstalled);
            } catch (Throwable error) {
                clearStateOnMain();
                postStatus(
                        "Ошибка установки модели:\n" + readable(error)
                );
            }
        }, "mobile-model-installer").start();
    }

    private void postStatus(String status) {
        activity.runOnUiThread(() -> listener.onStatus(status));
    }

    private void clearStateOnMain() {
        activity.runOnUiThread(this::clearState);
    }

    private void clearState() {
        activeId = -1L;
        activity.getSharedPreferences(
                PREFS,
                Context.MODE_PRIVATE
        ).edit().remove(PREF_ID).apply();
        notifyActive();
    }

    private void notifyActive() {
        listener.onActiveChanged(activeId >= 0);
    }

    private static String humanSize(long bytes) {
        if (bytes < 1024L * 1024L) {
            return (bytes / 1024L) + " КБ";
        }
        if (bytes < 1024L * 1024L * 1024L) {
            return String.format(
                    Locale.US,
                    "%.1f МБ",
                    bytes / (1024.0 * 1024.0)
            );
        }
        return String.format(
                Locale.US,
                "%.2f ГБ",
                bytes / (1024.0 * 1024.0 * 1024.0)
        );
    }

    private static String readable(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
                ? error.getClass().getSimpleName()
                : message;
    }

    public void close() {
        handler.removeCallbacks(poll);
        if (registered) {
            try {
                activity.unregisterReceiver(receiver);
            } catch (IllegalArgumentException ignored) {
            }
            registered = false;
        }
    }
}
