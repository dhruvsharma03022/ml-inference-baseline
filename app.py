from fastapi import FastAPI
import onnxruntime as ort
import numpy as np
import time

app = FastAPI()

# Load the model once when the container starts
session = ort.InferenceSession("mobilenetv2-7.onnx")

input_name = session.get_inputs()[0].name


@app.get("/")
def home():
    return {
        "status": "running",
        "model": "MobileNetV2",
        "runtime": "ONNX Runtime"
    }


@app.get("/predict")
def predict():
    # MobileNetV2 expects an image-shaped input.
    # For the baseline test, use a fixed synthetic input.
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    start = time.perf_counter()

    result = session.run(None, {
        input_name: input_data
    })

    latency = (time.perf_counter() - start) * 1000

    prediction = int(np.argmax(result[0]))

    return {
        "prediction": prediction,
        "latency_ms": latency
    }