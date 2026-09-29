package com.alex.aivideo;

/** Minimal IEEE-754 binary16 decoder used for Lite conditioning assets. */
public final class Fp16 {
    private Fp16() {}

    public static float toFloat(short bits) {
        int h = bits & 0xffff;
        int sign = (h >>> 15) & 0x1;
        int exponent = (h >>> 10) & 0x1f;
        int fraction = h & 0x3ff;

        int fSign = sign << 31;
        int fExponent;
        int fFraction;

        if (exponent == 0) {
            if (fraction == 0) {
                return Float.intBitsToFloat(fSign);
            }

            int e = -14;
            int frac = fraction;
            while ((frac & 0x400) == 0) {
                frac <<= 1;
                e--;
            }
            frac &= 0x3ff;
            fExponent = (e + 127) << 23;
            fFraction = frac << 13;
        } else if (exponent == 0x1f) {
            fExponent = 0xff << 23;
            fFraction = fraction << 13;
        } else {
            fExponent = (exponent - 15 + 127) << 23;
            fFraction = fraction << 13;
        }

        return Float.intBitsToFloat(fSign | fExponent | fFraction);
    }
}
