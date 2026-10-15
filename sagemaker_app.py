import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="SageMaker-compatible ONNX inference")

MODEL_PATH = Path(__file__).parent / "mobilenetv2-7.onnx"
session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name


@app.get("/ping")
def ping():
    return {"status": "healthy"}


@app.post("/invocations")
async def invocations(request: Request):
    start = time.perf_counter()

    try:
        body = await request.json()
        shape = body.get("shape", [1, 3, 224, 224])
        values = body.get("input")

        if values is None:
            x = np.random.rand(*shape).astype(np.float32)
        else:
            x = np.asarray(values, dtype=np.float32).reshape(shape)

        outputs = session.run(None, {input_name: x})
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "status": "ok",
            "latency_ms": round(elapsed_ms, 3),
            "output_shape": list(outputs[0].shape),
            "model": "MobileNetV2",
            "runtime": "ONNX Runtime"
        }
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)}
        )
