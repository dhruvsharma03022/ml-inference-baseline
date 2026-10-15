import requests
import time
import statistics

URL = "http://localhost:8000/predict"
NUM_REQUESTS = 100

latencies = []
successful = 0

for i in range(NUM_REQUESTS):
    start = time.perf_counter()

    response = requests.get(URL)

    end = time.perf_counter()

    if response.status_code == 200:
        successful += 1
        latencies.append((end - start) * 1000)

latencies.sort()

p50 = latencies[int(0.50 * len(latencies))]
p99 = latencies[int(0.99 * len(latencies))]

print("===== BASELINE RESULTS =====")
print(f"Total requests : {NUM_REQUESTS}")
print(f"Successful     : {successful}")
print(f"Failed         : {NUM_REQUESTS - successful}")
print(f"Mean latency   : {statistics.mean(latencies):.2f} ms")
print(f"P50 latency    : {p50:.2f} ms")
print(f"P99 latency    : {p99:.2f} ms")