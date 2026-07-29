# -*- coding: utf-8 -*-
"""§S13.19a: 캐리-EPS 공변 게이트를 4스프레드 전수에 재판정 (사용자 지시 확장).

재측정 없음 — §S13.19 산출물 `outputs/s13_19_carry_eps_preflight.csv`의
스프레드별 델타를 사전등록과 동일한 게이트 수식(G1/G2/G3)으로 판정만 한다.
**탐색적 4-way**: 통과 스프레드의 arm 채택은 별도 사전등록에 4-way 선택
사실 명시가 필요하다(결정 로그 §S13.19a 선언).

실행: <PY> scripts/preflight_s13_19a_all_spreads.py  (WD=ai_port, PYTHONPATH=.)
산출물: outputs/s13_19a_all_spreads.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.preflight_s13_19_carry_eps_covariation import judge_gates  # noqa: E402


def judge_row(row: pd.Series) -> dict:
    """CSV 한 행(스프레드)의 델타를 §S13.19 게이트로 판정."""
    return judge_gates(
        {"delta": float(row["gap_delta"])},
        {"H1": float(row["gap_delta_H1"]), "H2": float(row["gap_delta_H2"])},
        {"delta": float(row["spec_delta"])})


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    table = pd.read_csv(root / "outputs/s13_19_carry_eps_preflight.csv",
                        index_col="spread")
    rows = []
    for spread, row in table.iterrows():
        g = judge_row(row)
        rows.append({"spread": spread, **{k: row[k] for k in (
            "gap_bottom", "gap_mid", "gap_top", "gap_delta", "spec_delta",
            "gap_delta_H1", "gap_delta_H2")}, **g})
    out = pd.DataFrame(rows).set_index("spread")
    print("==== §S13.19a 캐리 공변 게이트 — 4스프레드 전수 (탐색적) ====")
    print(out.round(4).to_string())
    out.to_csv(root / "outputs/s13_19a_all_spreads.csv",
               encoding="utf-8-sig")
    print("\nsaved: outputs/s13_19a_all_spreads.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
