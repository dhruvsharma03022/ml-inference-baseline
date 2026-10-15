
import argparse
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config


def percentile(values, p):
    """Calculate a percentile using linear interpolation."""
    if not values:
        return float("nan")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def invoke_lambda(client, function_name):
    """Invoke Lambda once and measure client-observed latency."""
    start = time.perf_counter()

    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps({}).encode("utf-8"),
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    payload_bytes = response["Payload"].read()

    result = {
        "latency_ms": round(elapsed_ms, 3),
        "inference_latency_ms": None,
        "status_code": response.get("StatusCode"),
        "function_error": response.get("FunctionError", ""),
        "success": False,
        "error": "",
    }

    try:
        payload = json.loads(payload_bytes)

        if response.get("FunctionError"):
            result["error"] = json.dumps(payload)
        elif response.get("StatusCode") != 200:
            result["error"] = f"Lambda status: {response.get('StatusCode')}"
        else:
            # lambda_app.py returns a JSON object whose "body"
            # is itself a JSON-encoded string.
            body = payload.get("body", payload)

            if isinstance(body, str):
                body = json.loads(body)

            if isinstance(body, dict):
                inference_ms = body.get("latency_ms")
                if inference_ms is not None:
                    result["inference_latency_ms"] = float(inference_ms)

            result["success"] = True

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        result["error"] = f"Could not parse response: {exc}"

    return result


def summarize(label, values):
    if not values:
        print(f"{label}: no data")
        return

    print(f"\n{label}")
    print(f"  Count: {len(values)}")
    print(f"  Mean:  {statistics.mean(values):.3f} ms")
    print(f"  P50:   {percentile(values, 50):.3f} ms")
    print(f"  P99:   {percentile(values, 99):.3f} ms")
    print(f"  Min:   {min(values):.3f} ms")
    print(f"  Max:   {max(values):.3f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AWS Lambda ML inference latency"
    )
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--function", default="ml-inference-test")
    parser.add_argument("--region", default="eu-north-1")
    parser.add_argument("--output", default="lambda_benchmark.csv")
    args = parser.parse_args()

    if args.requests < 1 or args.warmup < 0:
        parser.error("--requests must be >= 1 and --warmup must be >= 0")

    client = boto3.client(
        "lambda",
        region_name=args.region,
        config=Config(
            connect_timeout=10,
            read_timeout=45,
            retries={"total_max_attempts": 1},
        ),
    )

    print(f"Function: {args.function}")
    print(f"Region:   {args.region}")
    print(f"Warm-up:  {args.warmup} requests")
    print(f"Measured: {args.requests} requests")
    print("\nRunning warm-up requests...")

    for i in range(args.warmup):
        result = invoke_lambda(client, args.function)
        print(
            f"  Warm-up {i + 1}/{args.warmup}: "
            f"{result['latency_ms']:.2f} ms, "
            f"success={result['success']}"
        )

    print("\nRunning measured requests...")
    rows = []
    benchmark_start = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()

    for i in range(args.requests):
        result = invoke_lambda(client, args.function)
        result["request_number"] = i + 1
        result["timestamp_utc"] = timestamp
        rows.append(result)

        if (i + 1) % 10 == 0 or i + 1 == args.requests:
            print(f"  Completed {i + 1}/{args.requests}")

    total_seconds = time.perf_counter() - benchmark_start

    successful = [r for r in rows if r["success"]]
    failed = [r for r in rows if not r["success"]]

    end_to_end = [r["latency_ms"] for r in successful]
    inference = [
        r["inference_latency_ms"]
        for r in successful
        if r["inference_latency_ms"] is not None
    ]

    print("\n========== LAMBDA BENCHMARK ==========")
    print(f"Function:             {args.function}")
    print(f"Requests attempted:   {len(rows)}")
    print(f"Successful:           {len(successful)}")
    print(f"Failed:               {len(failed)}")
    print(f"Total elapsed:        {total_seconds:.3f} seconds")
    print(f"Throughput:           {len(successful) / total_seconds:.3f} req/s")

    summarize("End-to-end latency (client-observed)", end_to_end)
    summarize("Model inference latency (handler-reported)", inference)

    output_path = Path(args.output)
    columns = [
        "request_number",
        "timestamp_utc",
        "success",
        "status_code",
        "function_error",
        "latency_ms",
        "inference_latency_ms",
        "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPer-request results saved to: {output_path.resolve()}")

    if failed:
        print("\nFirst failed request:")
        print(json.dumps(failed[0], indent=2))


if __name__ == "__main__":
    main()
