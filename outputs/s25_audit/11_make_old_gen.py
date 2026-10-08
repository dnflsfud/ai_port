"""Build the 'old generator' = working tree minus the 2026-10-07 call sites (F1 prefix masking, S25.1 cleaning).

HEAD (2026-08-26) is NOT the generator that produced the 2026-09-30 workbook (it lacks the September
completed_day_cutoff / 1BF2BF changes), so the pair baseline is the current tree with only the two
10-07 call sites neutralised. stale_earnings_tickers (warning only, no data effect) is left in place.
"""
import pathlib, sys

RS = pathlib.Path(r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study")

EDITS = {
    "create_universe_data.py": [(
        "            # §S25: price_v4 bfill 접두(관측 시작 전 미래값) 제거 — 환산 전.\n"
        "            df_subset = mask_sheet_prefix(df_subset, sheet_name)\n"
        "            # §S25.1: 센티먼트 범위 밖 센티널 NaN · 퇴화 OPER_MARGIN 열 제외 (접두 마스킹 뒤).\n"
        "            df_subset = clean_sheet_values(df_subset, sheet_name)\n",
        "            # [OLD-GEN PAIR BUILD] S25/S25.1 call sites removed for the s25_3 old-generator workbook.\n",
    )],
    "create_ai_signal_data.py": [(
        "    if sheet_name not in PREFIX_MASK_EXEMPT_SHEETS and available_tickers:\n"
        "        df_sheet[available_tickers] = mask_backfilled_prefix(\n"
        "            df_sheet[available_tickers].set_axis(df_sheet[\"date\"], axis=0)\n"
        "        ).to_numpy()\n",
        "    # [OLD-GEN PAIR BUILD] S25 prefix masking removed for the s25_3 old-generator workbook.\n",
    )],
}

for name, edits in EDITS.items():
    p = RS / name
    text = p.read_text(encoding="utf-8")
    for old, new in edits:
        assert text.count(old) == 1, f"{name}: expected exactly one match, got {text.count(old)}"
        text = text.replace(old, new)
    p.write_text(text, encoding="utf-8", newline="\n")
    print("old-gen edit applied:", name)
