"""Acceptance tests — Streamlit results dashboard pure helpers.

Written from the spec BEFORE implementation exists. The target module
`streamlit_app.py` must be refactored into UI-separated pure helpers
(`list_runs`, `load_metrics`, `load_result`) plus a `main()` guarded by
`if __name__ == "__main__": main()`, so that `import streamlit_app` does NOT
execute any Streamlit UI. Until that refactor lands EVERY test here MUST FAIL:
today `import streamlit_app` runs UI at module top-level (raising) and the
helper symbols do not exist. Any failure now (ImportError / AttributeError /
Streamlit runtime error) is the expected pre-implementation state.

Idioms (project convention): plain pytest functions, no fixtures; temp dirs via
`tempfile.TemporaryDirectory`; synthetic data only; fast. The target module is
imported INSIDE each test body so `pytest --collect-only` succeeds with zero
collection errors (a missing/side-effecting module surfaces as a runtime error
in the specific test, not a collection crash).

Note on numerics: this spec performs NO arithmetic — metric values are parsed
and passed through verbatim. The correct correctness gate is therefore exact
byte round-trip of the JSON numbers (asserted <0.5% tolerance == exact here),
NOT a recomputation. Expected values below are hardcoded independently of any
implementation logic.

======================================================================
합격기준(spec) ↔ 테스트 매핑표
----------------------------------------------------------------------
AC-1  import streamlit_app 부작용 없이 성공 + main 이 callable
        -> test_module_imports_without_ui
AC-2  list_runs: depth1(<out>/*/metrics.json) + depth2(<out>/*/*/metrics.json)
      스캔, label/metrics 파싱 정확
        -> test_list_runs_scans_depth1_and_2
AC-3  list_runs 정렬: 프로덕션 label 맨 앞, 나머지 알파벳순
        -> test_list_runs_production_first
AC-4  list_runs: 빈 디렉터리 -> [], 존재하지 않는 경로 -> []
        -> test_list_runs_empty_and_missing
AC-5  list_runs: 깨진 json 건너뜀(예외 전파 금지), 정상만 반환
        -> test_list_runs_skips_broken_json
AC-6  load_result: backtest_result.pkl 없으면 None
        -> test_load_result_none_when_missing
AC-7  load_metrics: metrics.json 전체(top-level 포함) 로드
        -> test_load_metrics_reads_full_json
AC-8  list_runs: json에 "label" 키 없으면 디렉터리명으로 대체
        -> test_list_runs_label_defaults_to_dirname
AC-9  list_runs: json에 "metrics" 없으면 metrics == {}
        -> test_list_runs_metrics_defaults_to_empty
AC-10 list_runs 원소 형태: {label, dir(Path), metrics, meta};
      meta 는 label/metrics 제외한 나머지 top-level 키
        -> test_list_runs_element_shape_and_meta
======================================================================
"""

import json
import pathlib
import tempfile

PROD_LABEL = "iter15_65tkr_reb21_vtg"  # canonical production run label (spec)


def _write_metrics(run_dir, *, label=None, metrics=None, extra=None):
    """Write a realistic metrics.json into `run_dir` and return its path.

    Mirrors the real on-disk shape:
        {"label": ..., "tuning_mode": "production",
         "metrics": {"information_ratio": ...}, "elapsed_sec": 1.0}
    `label` / `metrics` keys are omitted entirely when None so default-path
    behaviour (dirname fallback / empty metrics) can be exercised.
    """
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"tuning_mode": "production", "elapsed_sec": 1.0}
    if label is not None:
        payload["label"] = label
    if metrics is not None:
        payload["metrics"] = metrics
    if extra:
        payload.update(extra)
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# AC-1: module imports with NO UI side-effects; main() exists and is callable.
# ---------------------------------------------------------------------------
def test_module_imports_without_ui():
    import streamlit_app  # must not run any Streamlit UI at import time

    assert hasattr(streamlit_app, "main")
    assert callable(streamlit_app.main)


# ---------------------------------------------------------------------------
# AC-2: list_runs scans depth-1 and depth-2 metrics.json; parses label+metrics.
# ---------------------------------------------------------------------------
def test_list_runs_scans_depth1_and_2():
    from streamlit_app import list_runs

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        _write_metrics(out / "runA", label="alpha",
                       metrics={"information_ratio": 1.0})
        # depth-2: <out>/group/runB/metrics.json (group itself has no metrics)
        _write_metrics(out / "group" / "runB", label="beta",
                       metrics={"information_ratio": 0.5})

        runs = list_runs(out)

        assert len(runs) == 2
        by_label = {r["label"]: r for r in runs}
        assert set(by_label) == {"alpha", "beta"}
        # Metric numbers must round-trip verbatim (pass-through, exact).
        assert by_label["alpha"]["metrics"]["information_ratio"] == 1.0
        assert by_label["beta"]["metrics"]["information_ratio"] == 0.5
        # dir points at the run folder that actually holds metrics.json.
        assert isinstance(by_label["beta"]["dir"], pathlib.Path)
        assert (by_label["beta"]["dir"] / "metrics.json").exists()
        assert by_label["beta"]["dir"].name == "runB"


# ---------------------------------------------------------------------------
# AC-3: production label sorts first; the rest sort alphabetically by label.
# ---------------------------------------------------------------------------
def test_list_runs_production_first():
    from streamlit_app import list_runs

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        for name, label in [("r1", "zzz"), ("r2", PROD_LABEL), ("r3", "aaa")]:
            _write_metrics(out / name, label=label,
                           metrics={"information_ratio": 0.1})

        runs = list_runs(out)

        assert [r["label"] for r in runs] == [PROD_LABEL, "aaa", "zzz"]


# ---------------------------------------------------------------------------
# AC-4: empty dir -> []; non-existent path -> [] (no exception).
# ---------------------------------------------------------------------------
def test_list_runs_empty_and_missing():
    from streamlit_app import list_runs

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        assert list_runs(out) == []                       # empty directory
        assert list_runs(out / "does_not_exist") == []    # missing path


# ---------------------------------------------------------------------------
# AC-5: a broken JSON file is skipped; valid runs still returned; no raise.
# ---------------------------------------------------------------------------
def test_list_runs_skips_broken_json():
    from streamlit_app import list_runs

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        good = out / "good"
        _write_metrics(good, label="good_one",
                       metrics={"information_ratio": 2.0})
        bad = out / "bad"
        bad.mkdir(parents=True, exist_ok=True)
        (bad / "metrics.json").write_text("{not valid json,,,", encoding="utf-8")

        runs = list_runs(out)  # must not raise

        assert [r["label"] for r in runs] == ["good_one"]
        assert runs[0]["metrics"]["information_ratio"] == 2.0


# ---------------------------------------------------------------------------
# AC-6: load_result returns None when backtest_result.pkl is absent.
# ---------------------------------------------------------------------------
def test_load_result_none_when_missing():
    from streamlit_app import load_result

    with tempfile.TemporaryDirectory() as d:
        run_dir = pathlib.Path(d)  # no backtest_result.pkl present
        assert load_result(run_dir) is None


# ---------------------------------------------------------------------------
# AC-7: load_metrics loads the FULL metrics.json (top-level keys preserved).
# ---------------------------------------------------------------------------
def test_load_metrics_reads_full_json():
    from streamlit_app import load_metrics

    with tempfile.TemporaryDirectory() as d:
        run = pathlib.Path(d) / "run"
        path = _write_metrics(run, label="lbl",
                              metrics={"information_ratio": 1.4814375})

        loaded = load_metrics(path)

        assert loaded["label"] == "lbl"
        assert loaded["tuning_mode"] == "production"
        assert loaded["elapsed_sec"] == 1.0
        # nested metrics number preserved verbatim (exact pass-through).
        assert loaded["metrics"]["information_ratio"] == 1.4814375


# ---------------------------------------------------------------------------
# AC-8 (edge): missing "label" key -> label defaults to the directory name.
# ---------------------------------------------------------------------------
def test_list_runs_label_defaults_to_dirname():
    from streamlit_app import list_runs

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        _write_metrics(out / "my_run_dir", label=None,
                       metrics={"information_ratio": 0.3})

        runs = list_runs(out)

        assert len(runs) == 1
        assert runs[0]["label"] == "my_run_dir"


# ---------------------------------------------------------------------------
# AC-9 (edge): missing "metrics" key -> metrics defaults to {}.
# ---------------------------------------------------------------------------
def test_list_runs_metrics_defaults_to_empty():
    from streamlit_app import list_runs

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        _write_metrics(out / "no_metrics", label="nm", metrics=None)

        runs = list_runs(out)

        assert len(runs) == 1
        assert runs[0]["metrics"] == {}


# ---------------------------------------------------------------------------
# AC-10 (shape): element = {label, dir(Path), metrics, meta}; meta is the
# remaining top-level keys (label & metrics excluded).
# ---------------------------------------------------------------------------
def test_list_runs_element_shape_and_meta():
    from streamlit_app import list_runs

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        _write_metrics(out / "shape_run", label="m",
                       metrics={"information_ratio": 1.23})

        runs = list_runs(out)

        assert len(runs) == 1
        r = runs[0]
        assert set(r.keys()) >= {"label", "dir", "metrics", "meta"}
        assert isinstance(r["dir"], pathlib.Path)
        assert r["label"] == "m"
        assert r["metrics"]["information_ratio"] == 1.23
        meta = r["meta"]
        # meta carries the other top-level keys ...
        assert meta.get("tuning_mode") == "production"
        assert meta.get("elapsed_sec") == 1.0
        # ... but NOT the ones promoted to their own fields.
        assert "label" not in meta
        assert "metrics" not in meta
