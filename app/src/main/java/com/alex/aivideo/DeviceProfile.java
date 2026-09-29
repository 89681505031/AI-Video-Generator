package com.alex.aivideo;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Build;

import java.util.Locale;

public final class DeviceProfile {
    public final long totalRamGb;
    public final String soc;
    public final String abi;
    public final String profile;
    public final String note;

    private DeviceProfile(long totalRamGb, String soc, String abi, String profile, String note) {
        this.totalRamGb = totalRamGb;
        this.soc = soc;
        this.abi = abi;
        this.profile = profile;
        this.note = note;
    }

    public static DeviceProfile detect(Context context) {
        ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        am.getMemoryInfo(mi);

        long ramGb = Math.max(1L, Math.round(mi.totalMem / 1073741824.0));
        String abi = Build.SUPPORTED_ABIS.length > 0 ? Build.SUPPORTED_ABIS[0] : "unknown";
        String soc;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            soc = Build.SOC_MODEL == null || Build.SOC_MODEL.isBlank() ? Build.HARDWARE : Build.SOC_MODEL;
        } else {
            soc = Build.HARDWARE;
        }

        String profile;
        String note;
        if (ramGb >= 12) {
            profile = "MOBILE_HIGH";
            note = "Цель: короткие клипы с повышенным разрешением после установки совместимого model pack.";
        } else if (ramGb >= 8) {
            profile = "MOBILE_BALANCED";
            note = "Цель: короткие клипы в экономном режиме, с выгрузкой частей модели из памяти.";
        } else {
            profile = "MOBILE_LITE";
            note = "Нужна максимально облегчённая модель и очень короткие клипы.";
        }

        return new DeviceProfile(ramGb, soc, abi, profile, note);
    }

    public String summary() {
        return String.format(
                Locale.US,
                "RAM: ~%d GB\nSoC: %s\nABI: %s\nПрофиль: %s\n%s",
                totalRamGb, soc, abi, profile, note
        );
    }
}
