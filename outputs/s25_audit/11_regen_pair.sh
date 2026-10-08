#!/usr/bin/env bash
# §S25.3: regenerate the workbook twice from the same Bloomberg pull (Data/S&P500.xlsx 2026-10-02 16:10):
#   (old) working tree minus the 10-07 call sites -> ai_signal_data_oldgen_1002.xlsx / RL_Universe_Data_oldgen_1002.xlsx
#   (new) working-tree generator (F1 prefix masking + S25.1 rules) -> canonical ai_signal_data.xlsx
# Step 1 of run_data_pipeline.bat (news_sentiment_trend_analyzer.py, needs a Bloomberg session) is skipped:
# the existing Sentiment_Trend_Analysis.xlsx (2026-09-30 13:52) is consumed as-is by both generations.
set -u
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY="C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/Scripts/python.exe"
REPO="C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new"
RS="$REPO/machine/re_study"
SP="C:/Users/westl/AppData/Local/Temp/claude/C--Users-westl-PycharmProjects-pythonProject-venv-vf-new-machine-re-study-c2-ai-port/81b08b5c-3deb-40aa-9b69-086fce944fad/scratchpad"
BK="$SP/gen_backup"; mkdir -p "$BK"
FILES="universe_config.py create_universe_data.py create_ai_signal_data.py"
LOG="$SP/regen_pair.log"
ts() { date +%H:%M:%S; }
run_gen() {  # $1 = tag
  ( cd "$RS" && "$PY" -u create_universe_data.py ) > "$SP/gen_${1}_universe.log" 2>&1 || { echo "[$(ts)] FAIL create_universe_data ($1)"; return 1; }
  echo "[$(ts)] create_universe_data ($1) done"
  ( cd "$RS" && "$PY" -u create_ai_signal_data.py ) > "$SP/gen_${1}_signal.log" 2>&1 || { echo "[$(ts)] FAIL create_ai_signal_data ($1)"; return 1; }
  echo "[$(ts)] create_ai_signal_data ($1) done"
}
{
echo "[$(ts)] START pull=$(stat -c '%y %s' "$REPO/../Data/S&P500.xlsx")"
# 1. back up working-tree (new) generator files
for f in $FILES; do cp "$RS/$f" "$BK/$f"; done
sha256sum $(for f in $FILES; do echo "$BK/$f"; done) > "$BK/new.sha256"
# 2. old generator = working tree minus the two 10-07 call sites (HEAD 08-26 lacks the September changes)
PYTHONUTF8=1 "$PY" "$SP/make_old_gen.py" || { echo "[$(ts)] make_old_gen FAILED"; exit 1; }
grep -c "OLD-GEN PAIR BUILD" "$RS/create_universe_data.py" "$RS/create_ai_signal_data.py" | sed 's/^/  marker: /'
echo "[$(ts)] OLD generator in place"
# 3. old generation
if run_gen old; then
  mv -f "$RS/RL_Universe_Data.xlsx" "$RS/RL_Universe_Data_oldgen_1002.xlsx"
  mv -f "$RS/ai_signal_data.xlsx" "$RS/ai_signal_data_oldgen_1002.xlsx"
  echo "[$(ts)] OLD outputs renamed: $(stat -c '%y %s' "$RS/ai_signal_data_oldgen_1002.xlsx")"
else
  echo "[$(ts)] OLD generation failed -- canonical workbook untouched"
fi
# 4. restore working-tree (new) generator files and verify
for f in $FILES; do cp "$BK/$f" "$RS/$f"; done
( cd "$BK" && sha256sum -c new.sha256 --quiet ) && echo "[$(ts)] NEW generator restored (sha256 verified)" || echo "[$(ts)] RESTORE MISMATCH"
# 5. new generation -> canonical
if run_gen new; then
  echo "[$(ts)] NEW outputs: $(stat -c '%y %s' "$RS/ai_signal_data.xlsx") | RL $(stat -c '%y %s' "$RS/RL_Universe_Data.xlsx")"
  grep -a "경고\|WARN\|cutoff" "$SP/gen_new_universe.log" | head -20
else
  echo "[$(ts)] NEW generation failed"
fi
echo "[$(ts)] END"
} 2>&1 | tee "$LOG"
