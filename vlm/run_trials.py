"""Independent Claude odd-one-out trials. See VLM_README.md for the pilot workflow."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROBS = ("prob_a", "prob_b", "prob_c")
SCHEMA = {
    "type": "object",
    "properties": {**{key: {"type": "number"} for key in PROBS},
                   "explanation": {"type": "string"}},
    "required": [*PROBS, "explanation"],
    "additionalProperties": False,
}
FIELDS = ["trial_id", "id_a", "id_b", "id_c", "image_a", "image_b", "image_c",
          *PROBS, "explanation", "requested_model", "response_model", "settings_hash",
          "mode", "timestamp_utc", "request_id", "input_tokens", "output_tokens"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def local_path(value):
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def writable_path(value):
    path = local_path(value)
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("Output directory must be inside vlm/.")
    return path


def read_config(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if set(cfg) != {"model", "max_tokens", "temperature", "probability_tolerance",
                    "max_attempts", "timeout_seconds"}:
        raise ValueError("Config keys must match config.example.json exactly.")
    if not isinstance(cfg["model"], str) or not cfg["model"].strip():
        raise ValueError("model must be a nonempty string.")
    for key in ("max_tokens", "max_attempts"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key, low, high in (("temperature", 0, 1), ("probability_tolerance", 0, 0.01),
                           ("timeout_seconds", 0.001, 3600)):
        value = cfg[key]
        if key == "temperature" and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid {key}.")
    return cfg


def media_type(data):
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unsupported image signature (expected JPEG, PNG, GIF, or WebP).")


def load_trials(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"trial_id", "image_a", "image_b", "image_c"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {sorted(required)}.")
        trials, seen = [], set()
        for row in reader:
            trial_id = row["trial_id"].strip()
            if not trial_id or trial_id in seen:
                raise ValueError(f"Blank or duplicate trial_id: {trial_id!r}")
            seen.add(trial_id)
            trial = {"trial_id": trial_id}
            for label in "abc":
                raw = row[f"image_{label}"]
                if not raw or not raw.strip():
                    raise ValueError(f"Missing image_{label} in {trial_id}.")
                image = (path.parent / raw).resolve()
                data = image.read_bytes()
                media_type(data)
                if len(data) > 5 * 1024 * 1024:
                    raise ValueError(f"Image exceeds conservative 5 MiB limit: {image}")
                trial[f"image_{label}"] = str(image)
                trial[f"sha256_{label}"] = hashlib.sha256(data).hexdigest()
                trial[f"id_{label}"] = row.get(f"id_{label}", "") or ""
            trials.append(trial)
    if not trials:
        raise ValueError("Manifest is empty.")
    return trials


def validate_response(value, tolerance):
    if not isinstance(value, dict) or set(value) != {*PROBS, "explanation"}:
        raise ValueError("Response must contain exactly the four expected fields.")
    for key in PROBS:
        number = value[key]
        if type(number) not in (float, int) or not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"Invalid {key}.")
    if abs(sum(value[key] for key in PROBS) - 1) > tolerance:
        raise ValueError("Probabilities do not sum to 1 within tolerance.")
    if not isinstance(value["explanation"], str) or not value["explanation"].strip():
        raise ValueError("Explanation must be a nonempty string.")
    return value


def build_request(trial, cfg, prompt):
    content = []
    for label in "abc":
        data = Path(trial[f"image_{label}"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != trial[f"sha256_{label}"]:
            raise ValueError("Image changed since preflight.")
        content.extend([
            {"type": "text", "text": f"Image {label.upper()}:"},
            {"type": "image", "source": {"type": "base64", "media_type": media_type(data),
             "data": base64.b64encode(data).decode("ascii")}},
        ])
    content.append({"type": "text", "text": prompt})
    request = {"model": cfg["model"], "max_tokens": cfg["max_tokens"],
            "messages": [{"role": "user", "content": content}],
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}
    if cfg["temperature"] is not None:
        request["temperature"] = cfg["temperature"]
    return request


def validate_sdk_request(create, request):
    """Bind arguments locally; never calls the SDK method or sends a request."""
    try:
        inspect.signature(create).bind(**request)
    except TypeError as exc:
        raise ValueError(
            f"Request is incompatible with the installed Anthropic SDK: {exc}. "
            'For SDKs/models without temperature support, set "temperature": null in config.'
        ) from None


def append_json(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def completed_trials(path, settings_hash, tolerance, allowed_ids):
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Results CSV header is invalid.")
        done = set()
        for row in reader:
            if row["settings_hash"] != settings_hash or row["trial_id"] in done:
                raise ValueError("Results contain mixed settings or duplicate IDs.")
            if row["trial_id"] not in allowed_ids:
                raise ValueError("Results contain an unknown trial ID.")
            validate_response({**{key: float(row[key]) for key in PROBS},
                               "explanation": row["explanation"]}, tolerance)
            done.add(row["trial_id"])
    return done


def run(args):
    cfg = read_config(local_path(args.config))
    trials = load_trials(local_path(args.trials))
    prompt = local_path(args.prompt).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Prompt is empty.")
    selected = trials[:args.limit] if args.limit else trials
    if args.mode == "check":
        for trial in selected:
            build_request(trial, cfg, prompt)
        try:
            from anthropic import Anthropic
        except ImportError:
            print("Anthropic SDK is not installed; SDK compatibility check skipped.")
        else:
            # A dummy key permits local signature inspection without using credentials.
            with Anthropic(api_key="offline-preflight-only") as offline_client:
                validate_sdk_request(offline_client.messages.create,
                                     build_request(selected[0], cfg, prompt))
            print("Installed SDK accepts the request arguments (checked offline).")
        print(f"Preflight passed: {len(trials)} trials; {len(selected)} requests checked. No API calls.")
        return 0
    if args.mode == "live" and ("REPLACE" in cfg["model"] or cfg["model"].endswith("latest")):
        raise ValueError("Set an explicit model version in the config before a live run.")
    client = None
    sdk_version = None
    if args.mode == "live":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Set ANTHROPIC_API_KEY in the environment; never store it in config.")
        from anthropic import Anthropic
        sdk_version = importlib.metadata.version("anthropic")
        client = Anthropic(max_retries=0, timeout=cfg["timeout_seconds"])
        validate_sdk_request(client.messages.create, build_request(selected[0], cfg, prompt))
    settings = {"runner_version": 3, "mode": args.mode, "config": cfg, "prompt": prompt,
                "schema": SCHEMA, "trials": trials, "anthropic_sdk_version": sdk_version}
    settings_hash = digest(settings)
    out = writable_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two processes from appending duplicate trial results.
    lock = out / ".lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        return execute(out, settings, settings_hash, selected, cfg, prompt, client)
    finally:
        lock.unlink()


def execute(out, settings, settings_hash, trials, cfg, prompt, client):
    metadata = out / "settings.json"
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != settings:
            raise ValueError("Run inputs/settings changed. Use a new output directory.")
    else:
        if any((out / name).exists() for name in ("results.csv", "failures.jsonl", "responses.jsonl")):
            raise ValueError("Existing outputs have no settings snapshot. Use a new directory.")
        metadata.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    results = out / "results.csv"
    done = completed_trials(results, settings_hash, cfg["probability_tolerance"],
                            {trial["trial_id"] for trial in settings["trials"]})
    failed = 0
    for trial in trials:
        trial_id = trial["trial_id"]
        if trial_id in done:
            print(f"SKIP {trial_id}")
            continue
        for attempt in range(1, cfg["max_attempts"] + 1):
            try:
                request = build_request(trial, cfg, prompt)
                if client is None:
                    value = {"prob_a": 0.2, "prob_b": 0.3, "prob_c": 0.5,
                             "explanation": "MOCK: synthetic pipeline test; no image judgment."}
                    raw = {"mode": "mock", "value": value}
                    response_model, request_id, input_tokens, output_tokens = "mock", "", 0, 0
                else:
                    response = client.messages.create(**request)
                    raw = response.model_dump(mode="json")
                    # Save before validation, including refusals and truncated replies.
                    append_json(out / "responses.jsonl", {"trial_id": trial_id, "attempt": attempt,
                                "timestamp_utc": utc_now(), "response": raw})
                    if response.stop_reason != "end_turn":
                        raise ValueError(f"Unexpected stop reason: {response.stop_reason}")
                    value = json.loads("".join(block.text for block in response.content if block.type == "text"))
                    response_model = response.model
                    request_id = getattr(response, "_request_id", "") or ""
                    input_tokens, output_tokens = response.usage.input_tokens, response.usage.output_tokens
                if client is None:
                    append_json(out / "responses.jsonl", {"trial_id": trial_id, "attempt": attempt,
                                "timestamp_utc": utc_now(), "response": raw})
                validate_response(value, cfg["probability_tolerance"])
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                append_json(out / "failures.jsonl", {"trial_id": trial_id, "attempt": attempt,
                            "timestamp_utc": utc_now(), "error_type": type(exc).__name__,
                            "status_code": status,
                            "message": str(exc) if isinstance(exc, (ValueError, TypeError)) else "Request failed; inspect status/type."})
                if isinstance(exc, TypeError):
                    raise RuntimeError(f"Local SDK/programming error: {exc}; see failures.jsonl.") from None
                # Authentication, permissions, and malformed requests need intervention.
                if status in (400, 401, 403, 404, 413, 422):
                    raise RuntimeError(f"API rejected request (HTTP {status}); see failures.jsonl.") from None
                if attempt == cfg["max_attempts"]:
                    print(f"FAIL {trial_id} ({type(exc).__name__})")
                    failed += 1
                else:
                    time.sleep(min(2 ** (attempt - 1), 30))
                continue
            row = {key: trial.get(key, "") for key in FIELDS}
            row.update(value, requested_model=cfg["model"], response_model=response_model,
                       settings_hash=settings_hash, mode="live" if client else "mock",
                       timestamp_utc=utc_now(), request_id=request_id,
                       input_tokens=input_tokens, output_tokens=output_tokens)
            needs_header = not results.exists()
            with results.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
            print(f"OK {trial_id}")
            break
    print(f"Finished: {failed} failed trial(s). Outputs: {out}")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "mock", "live"), default="check")
    parser.add_argument("--trials", default="examples/demo_trials.csv")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--prompt", default="prompt.txt")
    parser.add_argument("--output", default="runs/mock-pilot")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
