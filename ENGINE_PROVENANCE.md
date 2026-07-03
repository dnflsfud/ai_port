# ENGINE_PROVENANCE — vendored snapshot

`./src` (그리고 `./run_variant.py`, `./scripts`, `./tests`, `./variants`)는
**cc2_rl 엔진의 벤더링 스냅샷**이다. 이 프로젝트(`ai_port`)는 이 스냅샷을
자체 소스로 보유하여 **cc2_rl 체크아웃 없이 단독 실행**된다 (`PYTHONPATH=CC2` 불필요).

| 항목 | 값 |
|---|---|
| 원본(source) | `…/machine/re_study/c2/ai_signal_cc2_rl` |
| 벤더링일 | 2026-06-19 |
| 레이아웃 | 미러(mirror) — 패키지명 `src` 유지 → `from src.X import …` 재작성 불필요 |
| 벤더링 LOC | ~8,600 (src + features + rl) |
| 원본 git | untracked (in-place 편집본, 커밋 해시 없음) |
| 포함된 수정 | 이번 세션의 P0~P3 + ultracode 리뷰 수정(M1~M3, L1~L8, GAP2/5) 반영본 |

## 독립성 경계
- **코드**: cc2_rl 자기경로 하드코딩 없음(유일한 매칭은 `src/rl/__init__.py` 주석). 검증: `PYTHONPATH=ai_port`만으로 34 테스트 통과, `src.__file__` → `ai_port/src`.
- **데이터**: `src/config.py:data_path`가 가리키는 **공유 Excel**(`…/re_study/ai_signal_data.xlsx`, 두 레포 밖)은 복사하지 않았다. 데이터 위치를 옮기려면 `data_path`만 수정.
- **산출물**: `save/load_checkpoint`·ablation 출력은 모두 **CWD 상대**(`./outputs`) → 이 폴더에 격리.

## 이제부터의 정본(canonical)
벤더링 시점부터 `ai_port/src`와 `cc2_rl/src`는 **독립적으로 발산**한다.
이 프로젝트의 엔진 수정은 **여기(`ai_port/src`)에서** 한다. cc2_rl 체크아웃을 고치지 않는다.

## 재동기화(원할 때만, 수동)
cc2_rl의 최신 엔진을 다시 끌어오려면(로컬 수정 덮어씀 주의):
```bash
CC2=…/c2/ai_signal_cc2_rl ; HERE=…/c2/ai_port
rm -rf "$HERE/src" && cp -r "$CC2/src" "$HERE/src"
# scripts/run_variant.py/tests/variants 도 동일하게. 이후 PYTHONPATH=$HERE 로 테스트.
```
