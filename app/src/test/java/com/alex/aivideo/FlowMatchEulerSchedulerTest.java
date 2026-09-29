package com.alex.aivideo;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class FlowMatchEulerSchedulerTest {
    @Test
    public void twoStepScheduleMatchesOfficialMobileI2VShift1() {
        FlowMatchEulerScheduler scheduler = new FlowMatchEulerScheduler(2);
        assertEquals(2, scheduler.steps());
        assertEquals(1.0f, scheduler.sigma(0), 1e-6f);
        assertEquals(0.001f, scheduler.sigma(1), 1e-6f);
        assertEquals(0.0f, scheduler.sigma(2), 1e-6f);
        assertEquals(1000.0f, scheduler.timestep(0), 1e-3f);
        assertEquals(1.0f, scheduler.timestep(1), 1e-3f);
    }

    @Test
    public void eulerStepUsesSigmaDelta() {
        FlowMatchEulerScheduler scheduler = new FlowMatchEulerScheduler(2);
        float[] sample = {1.0f, 2.0f};
        float[] model = {0.5f, -1.0f};
        scheduler.stepInPlace(sample, model, 0);
        float dt = (0.001f) - 1.0f;
        assertEquals(1.0f + dt * 0.5f, sample[0], 1e-6f);
        assertEquals(2.0f + dt * -1.0f, sample[1], 1e-6f);
    }

    @Test
    public void geometryMatchesOfficial512And720Profiles() {
        assertEquals(16, MobileI2VContract.latentWidth(512));
        assertEquals(16, MobileI2VContract.latentHeight(512));
        assertEquals(768, MobileI2VContract.sequencePositions(512, 512));
        assertEquals(40, MobileI2VContract.latentWidth(1280));
        assertEquals(23, MobileI2VContract.latentHeight(720));
        assertEquals(2760, MobileI2VContract.sequencePositions(1280, 720));
    }

    @Test
    public void conditionMaskPinsFirstTemporalSliceOnly() {
        float[] mask = MobileI2VContract.createConditionMask(512, 512);
        assertEquals(768, mask.length);
        int ones = 0;
        for (float value : mask) {
            if (value == 1.0f) ones++;
        }
        assertEquals(256, ones);
    }
}
