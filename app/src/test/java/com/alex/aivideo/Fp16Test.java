package com.alex.aivideo;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class Fp16Test {
    @Test
    public void decodesCommonHalfValues() {
        assertEquals(0.0f, Fp16.toFloat((short) 0x0000), 0.0f);
        assertEquals(1.0f, Fp16.toFloat((short) 0x3c00), 0.0f);
        assertEquals(-2.0f, Fp16.toFloat((short) 0xc000), 0.0f);
        assertEquals(65504.0f, Fp16.toFloat((short) 0x7bff), 0.0f);
    }

    @Test
    public void decodesInfinityAndNan() {
        assertTrue(Float.isInfinite(Fp16.toFloat((short) 0x7c00)));
        assertTrue(Float.isNaN(Fp16.toFloat((short) 0x7e00)));
    }
}
