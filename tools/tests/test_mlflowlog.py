import json
import sys
import urllib.request
from pathlib import Path

import pytest

from pianocv.mlflowlog import batches, flatten, main, model_metrics


def test_batches_split_large_metric_sets_and_add_tags_last() -> None:
    metrics = {f"m{i}": float(i) for i in range(1500)}
    bodies = batches("run", metrics, {"change": "camera-sim"}, 7)
    counts = [len(m) if isinstance(m := b.get("metrics"), list) else 0 for b in bodies]
    assert counts == [1000, 500, 0]
    assert bodies[-1]["tags"] == [{"key": "change", "value": "camera-sim"}]


def test_flatten_joins_paths_and_keeps_only_numbers() -> None:
    scores = {"frames": 40, "groups": {"gaps": {"recall": 0.9, "pck": {"0.1": 0.5}}}, "ok": True}
    assert flatten(scores, "real") == {
        "real/frames": 40.0,
        "real/groups/gaps/recall": 0.9,
        "real/groups/gaps/pck/0.1": 0.5,
    }


def test_model_metrics_picks_one_model_and_names_its_views() -> None:
    scores = {
        "keynet.a.onnx centroid": {"failure_rate": 0.3},
        "keynet.a.onnx centroid elevation low": {"failure_rate": 0.7},
        "keynet.b.onnx centroid": {"failure_rate": 0.1},
    }
    assert model_metrics(scores, "keynet.a.onnx", "synth-test") == {
        "synth-test/failure_rate": 0.3,
        "synth-test/elevation low/failure_rate": 0.7,
    }


def test_main_posts_batches_to_the_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"keynet.a.onnx centroid": {"failure_rate": 0.3}}))
    sent: list[tuple[str, str, str | None]] = []

    class Reply:
        def __enter__(self) -> "Reply":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def urlopen(request: urllib.request.Request) -> Reply:
        assert isinstance(request.data, bytes)
        sent.append((request.full_url, request.data.decode(), request.get_header("Host")))
        return Reply()

    monkeypatch.setattr("pianocv.mlflowlog.urllib.request.urlopen", urlopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlflowlog",
            "--tracking-uri",
            "http://localhost:5055",
            "--host",
            "mlflow.internal",
            "--run-id",
            "r1",
            "--scores",
            str(scores),
            "--model",
            "keynet.a.onnx",
            "--prefix",
            "synth",
            "--tag",
            "change=x",
        ],
    )
    main()
    assert [url for url, _, _ in sent] == [
        "http://localhost:5055/api/2.0/mlflow/runs/log-batch"
    ] * 2
    metric = json.loads(sent[0][1])["metrics"][0]
    assert (metric["key"], metric["value"]) == ("synth/failure_rate", 0.3)
    assert json.loads(sent[1][1])["tags"] == [{"key": "change", "value": "x"}]
    assert sent[0][2] == "mlflow.internal"


def test_main_refuses_a_non_http_tracking_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"x": 1}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlflowlog",
            "--tracking-uri",
            "file:///etc",
            "--run-id",
            "r",
            "--scores",
            str(scores),
            "--prefix",
            "p",
        ],
    )
    with pytest.raises(SystemExit):
        main()
