"""Log one model's scores from a score JSON onto its MLflow run, so every evaluation is stored
next to the model it scored."""

import argparse
import json
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path

Scores = Mapping[str, object]
# the tracking server accepts at most this many metrics in one batch
_BATCH = 1000


def flatten(scores: Scores, prefix: str) -> dict[str, float]:
    """Every number under `scores` as one metric, its path joined with slashes."""
    metrics: dict[str, float] = {}
    for key, value in scores.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            metrics |= flatten(value, name)
        elif isinstance(value, int | float) and not isinstance(value, bool):
            metrics[name] = float(value)
    return metrics


def model_metrics(scores: Scores, model: str, prefix: str) -> dict[str, float]:
    """The metrics of every entry for `model`, the part of each entry's name after the model and its
    decoder naming the view, as `evalkeynet --json` writes them."""
    metrics: dict[str, float] = {}
    for entry, value in scores.items():
        if not entry.startswith(model) or not isinstance(value, Mapping):
            continue
        view = entry[len(model) :].split(maxsplit=1)[1:] or [""]
        metrics |= flatten(value, "/".join(part for part in (prefix, view[0]) if part))
    return metrics


def batches(
    run_id: str, metrics: dict[str, float], tags: dict[str, str], now_ms: int
) -> list[dict[str, object]]:
    """The log-batch request bodies that record `metrics` and `tags` on the run."""
    items = [
        {"key": name.replace(" ", "_"), "value": value, "timestamp": now_ms, "step": 0}
        for name, value in metrics.items()
    ]
    bodies: list[dict[str, object]] = [
        {"run_id": run_id, "metrics": items[i : i + _BATCH]} for i in range(0, len(items), _BATCH)
    ]
    if tags:
        bodies.append({"run_id": run_id, "tags": [{"key": k, "value": v} for k, v in tags.items()]})
    return bodies


def main() -> None:
    parser = argparse.ArgumentParser(description="log a model's scores onto its MLflow run")
    parser.add_argument(
        "--tracking-uri", required=True, help="the MLflow server, e.g. a port-forward"
    )
    parser.add_argument("--host", help="the Host header the server expects, when it differs")
    parser.add_argument("--run-id", required=True, help="the MLflow run that trained the model")
    parser.add_argument("--scores", type=Path, required=True, help="a score JSON")
    parser.add_argument("--model", default="", help="the model file name its entries start with")
    parser.add_argument("--prefix", required=True, help="what was scored, e.g. synth-test or clips")
    parser.add_argument("--tag", action="append", default=[], help="key=value set on the run")
    args = parser.parse_args()
    scores = json.loads(args.scores.read_text())
    metrics = (
        model_metrics(scores, args.model, args.prefix)
        if args.model
        else flatten(scores, args.prefix)
    )
    if not metrics:
        raise SystemExit(f"no scores for {args.model or 'the file'} in {args.scores}")
    if urllib.parse.urlparse(args.tracking_uri).scheme not in ("http", "https"):
        raise SystemExit(f"the tracking URI must be http or https: {args.tracking_uri}")
    tags = dict(tag.partition("=")[::2] for tag in args.tag)
    headers = {"Content-Type": "application/json"} | ({"Host": args.host} if args.host else {})
    url = f"{args.tracking_uri.rstrip('/')}/api/2.0/mlflow/runs/log-batch"
    for body in batches(args.run_id, metrics, tags, int(time.time() * 1000)):
        request = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
        # the scheme is checked to be http or https above
        with urllib.request.urlopen(request) as response:  # nosec B310
            response.read()
    print(f"logged {len(metrics)} metrics and {len(tags)} tags to run {args.run_id}")


if __name__ == "__main__":
    main()
