
import json
import time
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("/var/task/mobilenetv2-7.onnx")
input_name = session.get_inputs()[0].name


def handler(event, context):
    input_data = np.random.rand(
        1, 3, 224, 224
    ).astype(np.float32)

    start = time.perf_counter()
    result = session.run(None, {input_name: input_data})
    latency_ms = (time.perf_counter() - start) * 1000

    body = {
        "prediction": int(np.argmax(result[0])),
        "latency_ms": round(latency_ms, 3),
        "model": "MobileNetV2",
        "runtime": "ONNX Runtime",
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
