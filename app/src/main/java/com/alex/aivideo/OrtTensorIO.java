package com.alex.aivideo;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;

final class OrtTensorIO {
    private OrtTensorIO() {}

    static OnnxTensor floatTensor(
            OrtEnvironment environment,
            float[] data,
            long[] shape
    ) throws Exception {
        ByteBuffer bytes = ByteBuffer
                .allocateDirect(data.length * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        FloatBuffer buffer = bytes.asFloatBuffer();
        buffer.put(data);
        buffer.rewind();
        return OnnxTensor.createTensor(environment, buffer, shape);
    }

    static OnnxTensor longTensor(
            OrtEnvironment environment,
            long[] data,
            long[] shape
    ) throws Exception {
        ByteBuffer bytes = ByteBuffer
                .allocateDirect(data.length * Long.BYTES)
                .order(ByteOrder.nativeOrder());
        LongBuffer buffer = bytes.asLongBuffer();
        buffer.put(data);
        buffer.rewind();
        return OnnxTensor.createTensor(environment, buffer, shape);
    }

    static float[] copyFloatOutput(OnnxValue value, int expectedElements)
            throws Exception {
        if (!(value instanceof OnnxTensor)) {
            throw new IllegalArgumentException("ONNX output не является tensor.");
        }

        FloatBuffer buffer = ((OnnxTensor) value).getFloatBuffer();
        if (buffer == null) {
            throw new IllegalArgumentException(
                    "ONNX output нельзя безопасно прочитать как float."
            );
        }

        buffer.rewind();
        if (buffer.remaining() != expectedElements) {
            throw new IllegalArgumentException(
                    "Неверный размер ONNX output: "
                            + buffer.remaining() + ", ожидалось " + expectedElements + "."
            );
        }

        float[] result = new float[expectedElements];
        buffer.get(result);
        return result;
    }
}
