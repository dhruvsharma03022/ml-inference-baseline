import csv
import time
import statistics
from datetime import datetime, timezone

import requests

URL = "http://localhost:8000/predict"
NUM_REQUESTS = 100
WARMUP_REQUESTS = 5
TIMEOUT_SECONDS = 30

def percentile(values, p):
    values = sorted(values)
    index = max(0, min(len(values) - 1, int((p * len(values) + 0.999999) - 1)))
    return values[index]

session = requests.Session()

# Warm-up requests are excluded from the measurements.
for _ in range(WARMUP_REQUESTS):
    try:
        session.get(URL, timeout=TIMEOUT_SECONDS).raise_for_status()
    except requests.RequestException as exc:
        print(f"Warm-up request failed: {exc}")
        print("Check the service with: docker logs ml-baseline")
        raise SystemExit(1)

results = []

for i in range(1, NUM_REQUESTS + 1):
    timestamp = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()

    try:
        response = session.get(URL, timeout=TIMEOUT_SECONDS)
        elapsed_ms = (time.perf_counter() - start) * 1000
        success = response.status_code == 200
        error = "" if success else f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        success = False
        error = str(exc)

    results.append({
        "request_number": i,
        "timestamp_utc": timestamp,
        "success": success,
        "latency_ms": round(elapsed_ms, 3),
        "error": error,
    })

with open("docker_benchmark.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

latencies = sorted(r["latency_ms"] for r in results if r["success"])
successful = len(latencies)

print("===== LOCAL DOCKER BASELINE RESULTS =====")
print(f"Total requests : {NUM_REQUESTS}")
print(f"Successful     : {successful}")
print(f"Failed         : {NUM_REQUESTS - successful}")

if latencies:
    print(f"Mean latency   : {statistics.mean(latencies):.2f} ms")
    print(f"P50 latency    : {percentile(latencies, 0.50):.2f} ms")
    print(f"P95 latency    : {percentile(latencies, 0.95):.2f} ms")
    print(f"P99 latency    : {percentile(latencies, 0.99):.2f} ms")
else:
    print("No successful requests; inspect the container and logs.")

print("CSV saved      : docker_benchmark.csv")
