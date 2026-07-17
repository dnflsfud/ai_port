"""Build a Korean PDF explaining the portfolio and BAT orchestration."""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "portfolio_logic_and_bat_process.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E6F9E")
LIGHT_BLUE = colors.HexColor("#EAF3F8")
TEAL = colors.HexColor("#2A7F7F")
LIGHT_TEAL = colors.HexColor("#E8F4F2")
GOLD = colors.HexColor("#D6A33B")
LIGHT_GOLD = colors.HexColor("#FFF6DE")
RED = colors.HexColor("#B84A4A")
LIGHT_RED = colors.HexColor("#FBECEC")
INK = colors.HexColor("#24313C")
MUTED = colors.HexColor("#5F6F7A")
GRID = colors.HexColor("#CAD6DE")
PAPER = colors.HexColor("#F8FAFB")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Malgun", r"C:\Windows\Fonts\malgun.ttf"))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", r"C:\Windows\Fonts\malgunbd.ttf"))


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Malgun-Bold", fontSize=25,
            leading=34, textColor=NAVY, alignment=TA_LEFT, spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Malgun", fontSize=12,
            leading=19, textColor=MUTED, spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Malgun-Bold", fontSize=17,
            leading=24, textColor=NAVY, spaceBefore=2, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Malgun-Bold", fontSize=12.5,
            leading=18, textColor=BLUE, spaceBefore=7, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName="Malgun", fontSize=9.4,
            leading=15.2, textColor=INK, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], fontName="Malgun", fontSize=8,
            leading=12, textColor=MUTED, spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "callout", parent=base["BodyText"], fontName="Malgun-Bold", fontSize=10,
            leading=16, textColor=NAVY, leftIndent=7, rightIndent=7,
            borderColor=GOLD, borderWidth=1, borderPadding=8,
            backColor=LIGHT_GOLD, spaceBefore=5, spaceAfter=10,
        ),
        "table": ParagraphStyle(
            "table", parent=base["BodyText"], fontName="Malgun", fontSize=7.6,
            leading=11, textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "table_head", parent=base["BodyText"], fontName="Malgun-Bold", fontSize=8,
            leading=11, textColor=colors.white, alignment=TA_CENTER,
        ),
        "center": ParagraphStyle(
            "center", parent=base["BodyText"], fontName="Malgun", fontSize=8,
            leading=11, textColor=INK, alignment=TA_CENTER,
        ),
    }


def P(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def table(data, widths, st, header=True, row_bgs=None):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Malgun"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Malgun-Bold"),
        ]
        for r in range(1, len(data)):
            cmds.append(("BACKGROUND", (0, r), (-1, r), PAPER if r % 2 else colors.white))
    if row_bgs:
        for r, bg in row_bgs.items():
            cmds.append(("BACKGROUND", (0, r), (-1, r), bg))
    t.setStyle(TableStyle(cmds))
    return t


def flow_diagram(nodes, width=174 * mm, box_h=18 * mm, gap=5 * mm, cols=4):
    """Create a wrapped, arrow-connected flow diagram."""
    rows = (len(nodes) + cols - 1) // cols
    row_gap = 12 * mm
    height = rows * box_h + max(0, rows - 1) * row_gap + 3 * mm
    d = Drawing(width, height)
    box_w = (width - (cols - 1) * gap) / cols
    positions = []
    for idx, node in enumerate(nodes):
        row = idx // cols
        pos = idx % cols
        reverse = row % 2 == 1
        col = (cols - 1 - pos) if reverse else pos
        x = col * (box_w + gap)
        y = height - (row + 1) * box_h - row * row_gap
        fill = node.get("fill", LIGHT_BLUE)
        stroke = node.get("stroke", BLUE)
        d.add(Rect(x, y, box_w, box_h, rx=4, ry=4, fillColor=fill,
                   strokeColor=stroke, strokeWidth=1.1))
        lines = node["label"].split("\n")
        leading = 10
        start_y = y + box_h / 2 + (len(lines) - 1) * leading / 2 - 3
        for j, line in enumerate(lines):
            d.add(String(x + box_w / 2, start_y - j * leading, line,
                         fontName="Malgun-Bold" if j == 0 else "Malgun",
                         fontSize=7.4 if j == 0 else 6.5,
                         fillColor=INK, textAnchor="middle"))
        positions.append((x, y, box_w, box_h, row, col))

    for idx in range(len(nodes) - 1):
        x, y, bw, bh, row, col = positions[idx]
        nx, ny, nbw, nbh, nrow, ncol = positions[idx + 1]
        if row == nrow:
            if nx > x:
                x1, y1, x2, y2 = x + bw, y + bh / 2, nx, ny + nbh / 2
            else:
                x1, y1, x2, y2 = x, y + bh / 2, nx + nbw, ny + nbh / 2
        else:
            x1, y1 = x + bw / 2, y
            x2, y2 = nx + nbw / 2, ny + nbh
        d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=1.0))
        angle = 3.2
        if abs(x2 - x1) > abs(y2 - y1):
            if x2 > x1:
                pts = [x2, y2, x2 - angle, y2 + 2, x2 - angle, y2 - 2]
            else:
                pts = [x2, y2, x2 + angle, y2 + 2, x2 + angle, y2 - 2]
        else:
            pts = [x2, y2, x2 - 2, y2 + angle, x2 + 2, y2 + angle]
        d.add(Polygon(pts, fillColor=MUTED, strokeColor=MUTED))
    return d


def cover_page(st):
    story = [Spacer(1, 20 * mm)]
    story.append(P("AI Portfolio Logic & BAT Process", st["title"]))
    story.append(P("포트폴리오 신호 생성, 위험 제약 최적화, 운영 산출물, 자동화 프로세스 정리", st["subtitle"]))
    story.append(Spacer(1, 5 * mm))
    story.append(flow_diagram([
        {"label": "DATA\nExcel & Universe"},
        {"label": "SIGNAL\nFeatures & LightGBM"},
        {"label": "PORTFOLIO\nMVO & Execution", "fill": LIGHT_TEAL, "stroke": TEAL},
        {"label": "OPERATIONS\nBundle & Dashboard", "fill": LIGHT_GOLD, "stroke": GOLD},
    ], cols=4, box_h=23 * mm))
    story.append(Spacer(1, 12 * mm))
    story.append(P(
        "이 시스템은 65종목의 AI 점수를 바로 투자 비중으로 사용하지 않는다. "
        "신호를 생성한 뒤 시가총액 벤치마크 대비 Tracking Error, 종목 편차, 섹터 편차, "
        "회전율을 함께 제한해 운용 가능한 Enhanced Index 포트폴리오로 변환한다.",
        st["callout"],
    ))
    facts = [
        [P("현재 역할", st["table_head"]), P("모델", st["table_head"]), P("리밸런싱", st["table_head"]), P("벤치마크", st["table_head"])],
        [P("Causal Rank 65<br/><b>PRODUCTION</b>", st["center"]), P("횡단면 Rank LightGBM", st["center"]), P("21 거래일", st["center"]), P("시가총액 가중", st["center"])],
        [P("Legacy S0<br/><b>CHALLENGER</b>", st["center"]), P("Regression LightGBM", st["center"]), P("21 거래일", st["center"]), P("시가총액 가중", st["center"])],
    ]
    story.append(table(facts, [42 * mm, 49 * mm, 38 * mm, 43 * mm], st))
    story.append(Spacer(1, 23 * mm))
    story.append(P("작성 기준: 2026-07-13 | 저장소: ai_port", st["small"]))
    story.append(PageBreak())
    return story


def architecture_page(st):
    story = [P("1. 전체 아키텍처", st["h1"])]
    nodes = [
        {"label": "원천 데이터\n가격·수익률·시총·실적"},
        {"label": "UniverseData\n정렬·결측·65종목"},
        {"label": "Feature Engine\nCore feature panel"},
        {"label": "Target Engine\n20일 specific return"},
        {"label": "Walk-forward Model\nRegression / Rank"},
        {"label": "Signal Overlay\nEMA·PEAD·Growth·VTG"},
        {"label": "MVO\nBM·공분산·제약", "fill": LIGHT_TEAL, "stroke": TEAL},
        {"label": "Execution\n부분 체결·재투영", "fill": LIGHT_TEAL, "stroke": TEAL},
        {"label": "Backtest\nPnL·비용·Drift"},
        {"label": "Operating Bundle\n성과·위험·보유·기여"},
        {"label": "Validation\nHash·날짜·유니버스"},
        {"label": "Dashboard\n9개 운영 탭", "fill": LIGHT_GOLD, "stroke": GOLD},
    ]
    story.append(flow_diagram(nodes, cols=4, box_h=18 * mm))
    story.append(Spacer(1, 5 * mm))
    data = [
        [P("단계", st["table_head"]), P("구현 위치", st["table_head"]), P("핵심 책임", st["table_head"])],
        [P("데이터", st["table"]), P("src/data_loader.py", st["table"]), P("Excel 시트 로딩, 필수 시트 기반 유니버스, raw return 보존, 데이터 품질 진단", st["table"])],
        [P("피처", st["table"]), P("src/features/assembly.py", st["table"]), P("회계·가격·리비전·팩터·레짐·매크로 피처 조립 및 core whitelist 적용", st["table"])],
        [P("타깃", st["table"]), P("src/target_engine.py", st["table"]), P("PCA 공통요인을 제거한 20일 선행 specific return 생성", st["table"])],
        [P("모델", st["table"]), P("src/model_trainer.py", st["table"]), P("5년 학습창의 walk-forward 학습, EMA, 퇴화 모델 감시", st["table"])],
        [P("포트폴리오", st["table"]), P("src/portfolio_optimizer.py", st["table"]), P("예상수익, active risk, turnover를 결합한 ECOS MVO", st["table"])],
        [P("시뮬레이션", st["table"]), P("src/backtest.py", st["table"]), P("신호 오버레이, 1일 지연, 21일 리밸런싱, 거래비용, 결과 집계", st["table"])],
        [P("운영 화면", st["table"]), P("streamlit_app.py", st["table"]), P("운영 bundle과 registry를 읽어 성과·위험·기여·비교 화면 제공", st["table"])],
    ]
    story.append(table(data, [27 * mm, 47 * mm, 98 * mm], st))
    story.append(PageBreak())
    return story


def signal_page(st):
    story = [P("2. 신호 생성과 모델 경로", st["h1"])]
    story.append(flow_diagram([
        {"label": "Core Features\n회계·가격·Sellside"},
        {"label": "LightGBM Raw Score\n회귀 또는 Rank"},
        {"label": "Prediction EMA\nalpha = 0.5"},
        {"label": "PEAD Boost\n실적 발표 후 리비전"},
        {"label": "Growth Tilt\n성장·리비전"},
        {"label": "Value-trap Gate\n저평가+약한 모멘텀"},
        {"label": "1 Day Signal Lag\n동일 종가 체결 방지", "fill": LIGHT_GOLD, "stroke": GOLD},
        {"label": "MVO Expected Return\n종목별 mu 입력", "fill": LIGHT_TEAL, "stroke": TEAL},
    ], cols=4, box_h=19 * mm))
    story.append(P("주요 파라미터", st["h2"]))
    params = [
        [P("영역", st["table_head"]), P("현재 설정", st["table_head"]), P("해석", st["table_head"])],
        [P("유니버스", st["table"]), P("65종목", st["table"]), P("필수 데이터 시트의 공통 종목으로 고정", st["table"])],
        [P("타깃", st["table"]), P("20일 forward, PCA 2개 요인 제거", st["table"]), P("시장 공통 움직임보다 종목 고유 수익을 학습", st["table"])],
        [P("학습", st["table"]), P("1260일 train / 126일 val", st["table"]), P("장기 학습창과 별도 검증창 사용", st["table"])],
        [P("재학습", st["table"]), P("63거래일", st["table"]), P("분기 수준으로 모델 갱신", st["table"])],
        [P("리밸런싱", st["table"]), P("21거래일", st["table"]), P("모델 재학습보다 자주 비중을 재산출", st["table"])],
        [P("거래비용", st["table"]), P("편도 10bp", st["table"]), P("리밸런싱 시 L1 turnover에 적용", st["table"])],
    ]
    story.append(table(params, [30 * mm, 58 * mm, 84 * mm], st))
    story.append(P("두 모델의 관계", st["h2"]))
    roles = [
        [P("포트폴리오", st["table_head"]), P("역할", st["table_head"]), P("의도적인 차이", st["table_head"]), P("공통 조건", st["table_head"])],
        [P("Causal Rank 65", st["table"]), P("Production", st["table"]), P("날짜별 rank_xendcg 횡단면 순위 학습", st["table"]), P("유니버스·오버레이·MVO·비용 동일", st["table"])],
        [P("Legacy S0", st["table"]), P("Challenger", st["table"]), P("LightGBM regression으로 specific return 예측", st["table"]), P("유니버스·오버레이·MVO·비용 동일", st["table"])],
    ]
    story.append(table(roles, [37 * mm, 27 * mm, 58 * mm, 50 * mm], st))
    story.append(P(
        "역할의 기준은 Variant와 outputs/portfolio_registry.json이다. BAT 주석과 로그도 이 기준으로 통일한다.",
        st["callout"],
    ))
    story.append(PageBreak())
    return story


def optimizer_page(st):
    story = [P("3. MVO와 리밸런싱 실행", st["h1"])]
    story.append(P(
        "목적함수: 모델 점수의 가중합을 높이되, 벤치마크 대비 active variance, 이전 비중 대비 turnover, "
        "선택적인 factor exposure를 페널티로 차감한다.", st["body"]
    ))
    formula = Table([[P(
        "maximize  mu' w  -  risk_aversion * (w-b)' Sigma (w-b)  -  turnover_penalty * |w-w_prev|  -  factor_penalty",
        st["center"]
    )]], colWidths=[172 * mm])
    formula.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_TEAL),
        ("BOX", (0, 0), (-1, -1), 1, TEAL),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(formula)
    story.append(Spacer(1, 5 * mm))
    constraints = [
        [P("제약", st["table_head"]), P("값", st["table_head"]), P("운용 의미", st["table_head"])],
        [P("총 비중 / 공매도", st["table"]), P("100% / 금지", st["table"]), P("Fully invested long-only", st["table"])],
        [P("종목 최대 비중", st["table"]), P("15%", st["table"]), P("절대 보유비중 상한", st["table"])],
        [P("종목 active 편차", st["table"]), P("+/- 4%", st["table"]), P("BM 대비 개별 종목 베팅 제한", st["table"])],
        [P("L1 active share", st["table"]), P("45%", st["table"]), P("전체 active budget 상한", st["table"])],
        [P("연율 Tracking Error", st["table"]), P("4.5% 이하", st["table"]), P("벤치마크 대비 변동성 예산", st["table"])],
        [P("1회 turnover", st["table"]), P("15% 이하", st["table"]), P("리밸런싱별 거래량 제한", st["table"])],
        [P("섹터 편차", st["table"]), P("+/- 10%", st["table"]), P("섹터 집중 리스크 제한", st["table"])],
        [P("Score-gated OW", st["table"]), P("점수 > 0", st["table"]), P("비양수 신호 종목의 overweight 차단", st["table"])],
    ]
    story.append(table(constraints, [43 * mm, 34 * mm, 95 * mm], st))
    story.append(P("리밸런싱일 처리 순서", st["h2"]))
    story.append(flow_diagram([
        {"label": "1. 당일 PnL\n진입 비중 사용"},
        {"label": "2. Weight Drift\n당일 수익 반영"},
        {"label": "3. Close Rebalance\nMVO 목표 계산"},
        {"label": "4. Dynamic Execution\n확신도·No-trade"},
        {"label": "5. Projection\nHard constraint 복원", "fill": LIGHT_TEAL, "stroke": TEAL},
        {"label": "6. 거래비용\n당일 PnL 차감"},
        {"label": "7. 다음 거래일\n새 비중 효력 발생", "fill": LIGHT_GOLD, "stroke": GOLD},
    ], cols=4, box_h=18 * mm))
    story.append(P(
        "ECOS 해가 없거나 non-finite이면 benchmark book으로 fallback하고, 실패 횟수·원인·solver 사용 통계를 결과에 기록한다.",
        st["callout"],
    ))
    story.append(PageBreak())
    return story


def operating_page(st):
    story = [P("4. 연구 게이트와 운영 산출물", st["h1"])]
    story.append(P("Pictet adoption 연구 단계", st["h2"]))
    story.append(flow_diagram([
        {"label": "Stage 0\nS0 ECOS baseline"},
        {"label": "Stage 1\nAlpha attribution"},
        {"label": "Stage 2\nOverlay 2^3 ablation"},
        {"label": "Stage 3\nFactor-neutral ablation"},
        {"label": "Stage 4\nSelection bias / DSR"},
        {"label": "Verdict\nadoption_summary.json", "fill": LIGHT_GOLD, "stroke": GOLD},
    ], cols=3, box_h=19 * mm))
    story.append(P("운영 Bundle", st["h2"]))
    bundles = [
        [P("파일", st["table_head"]), P("내용", st["table_head"]), P("Dashboard 사용처", st["table_head"])],
        [P("portfolio.json", st["table"]), P("역할, 모델, 기준일, 유니버스, source hash", st["table"]), P("포트폴리오 식별·신선도", st["table"])],
        [P("performance.json / returns.csv", st["table"]), P("성과 지표와 일별 수익률", st["table"]), P("Overview·Performance·Comparison", st["table"])],
        [P("holdings.json", st["table"]), P("최신 비중, BM 비중, active, 집중도", st["table"]), P("Risk·Turnover·Comparison", st["table"])],
        [P("risk.json / contribution.json", st["table"]), P("종목·섹터 위험 및 성과 기여", st["table"]), P("Risk·Contribution", st["table"])],
        [P("operations.json / monitoring.json", st["table"]), P("거래 목록, 회전율, 리밸런싱 일정, rolling 지표", st["table"]), P("Turnover·Monitoring", st["table"])],
        [P("features.json / feature_attribution.json", st["table"]), P("모델 중요도와 종목별 SHAP driver", st["table"]), P("Signals & Gates·Stock Drivers", st["table"])],
    ]
    story.append(table(bundles, [49 * mm, 76 * mm, 47 * mm], st))
    story.append(P("Bundle Validator가 확인하는 항목", st["h2"]))
    checks = [
        [P("날짜 일치", st["center"]), P("유니버스 Hash", st["center"]), P("Source Metrics Hash", st["center"]), P("리밸런싱 일정", st["center"])],
        [P("returns / performance / portfolio", st["small"]), P("두 포트폴리오의 종목 동일성", st["small"]), P("export 후 원본 변경 탐지", st["small"]), P("last / next / row counter", st["small"])],
    ]
    story.append(table(checks, [43 * mm] * 4, st, header=False, row_bgs={0: LIGHT_BLUE, 1: colors.white}))
    story.append(P(
        "검증을 통과한 두 bundle만 outputs/portfolio_registry.json에 게시되고, Streamlit은 registry의 portfolio_role을 기준으로 production과 challenger를 선택한다.",
        st["callout"],
    ))
    story.append(PageBreak())
    return story


def dashboard_bat_page(st):
    story = [P("5. run_dashboard.bat", st["h1"])]
    story.append(P(
        "원천 데이터 확인부터 연구 평가, 두 포트폴리오 운영 bundle, 검증, Dashboard 실행까지 순차 수행하는 foreground 파이프라인이다.",
        st["body"],
    ))
    nodes = [
        {"label": "0. Preflight\nPython·Excel 확인"},
        {"label": "1. Adoption\nS0·Attribution·Ablation·DSR"},
        {"label": "2. Data Quality\ncoverage·staleness"},
        {"label": "3. Legacy S0\nCHALLENGER export", "fill": LIGHT_GOLD, "stroke": GOLD},
        {"label": "4. Causal Rank 65\nPRODUCTION backtest", "fill": LIGHT_TEAL, "stroke": TEAL},
        {"label": "5. Causal Rank 65\nPRODUCTION export", "fill": LIGHT_TEAL, "stroke": TEAL},
        {"label": "6. Validate\n두 bundle + registry"},
        {"label": "7. Streamlit\nCtrl+C까지 blocking"},
    ]
    story.append(flow_diagram(nodes, cols=4, box_h=21 * mm))
    story.append(Spacer(1, 4 * mm))
    details = [
        [P("단계", st["table_head"]), P("실행 명령", st["table_head"]), P("실패 시", st["table_head"])],
        [P("1", st["table"]), P("run_pictet_adoption.py", st["table"]), P("즉시 :fail", st["table"])],
        [P("3", st["table"]), P("scripts/export_operating_data.py", st["table"]), P("Legacy S0 export 중단", st["table"])],
        [P("4", st["table"]), P("run_variant.py --variant variants/codex_causal_rank_65.yaml --no-cache", st["table"]), P("Production backtest 중단", st["table"])],
        [P("5", st["table"]), P("export_operating_data.py --variant ... --operating-dir outputs/operating_codex_causal_rank_65", st["table"]), P("Production export 중단", st["table"])],
        [P("6", st["table"]), P("validate_portfolio_bundles.py --bundle ...", st["table"]), P("Registry 게시 중단", st["table"])],
        [P("7", st["table"]), P("python -m streamlit run streamlit_app.py", st["table"]), P("Dashboard 오류 후 :fail", st["table"])],
    ]
    story.append(table(details, [18 * mm, 121 * mm, 33 * mm], st))
    story.append(P("동작 특성", st["h2"]))
    story.append(P(
        "- 각 단계 직후 ERRORLEVEL을 검사하므로 fail-fast 방식이다.<br/>"
        "- Streamlit은 현재 콘솔을 점유하며 Ctrl+C까지 종료되지 않는다.<br/>"
        "- 계산 결과가 이미 있으면 Dashboard만 별도로 실행하는 편이 효율적이다.<br/>"
        "- 전체 Adoption과 no-cache Production run을 포함하므로 단순 화면 실행 용도로는 무겁다.",
        st["body"],
    ))
    story.append(P(
        "명칭 반영: BAT 로그에서 Legacy S0는 challenger, Causal Rank 65는 production으로 표시한다.",
        st["callout"],
    ))
    story.append(PageBreak())
    return story


def upload_bat_page(st):
    story = [P("6. run_and_upload.bat와 Scheduler", st["h1"])]
    story.append(P("run_and_upload.bat", st["h2"]))
    story.append(flow_diagram([
        {"label": "1. Environment\nPython·cvxpy·ECOS"},
        {"label": "2. Tests\npytest tests -q"},
        {"label": "3-4. Legacy S0\nCHALLENGER run + export", "fill": LIGHT_GOLD, "stroke": GOLD},
        {"label": "5-6. Causal Rank 65\nPRODUCTION run + export", "fill": LIGHT_TEAL, "stroke": TEAL},
        {"label": "7. Validate\nregistry 게시"},
        {"label": "8. Git Commit\ngit add -A"},
        {"label": "9. Git Sync\nfetch·pull --rebase·push"},
        {"label": "10. Dashboard\n환경변수로 생략 가능"},
    ], cols=4, box_h=21 * mm))
    story.append(P("예약 실행 Wrapper", st["h2"]))
    story.append(flow_diagram([
        {"label": "AI_PORT_NO_DASHBOARD=1\n화면 실행 금지"},
        {"label": "logs 폴더 준비\n없으면 생성"},
        {"label": "run_and_upload.bat\nscheduled: weekday run"},
        {"label": "scheduled_run_last.log\nstdout·stderr 덮어쓰기", "fill": LIGHT_GOLD, "stroke": GOLD},
    ], cols=4, box_h=20 * mm))
    story.append(P("운영 시 주의사항", st["h2"]))
    cautions = [
        [P("항목", st["table_head"]), P("현재 동작", st["table_head"]), P("권장 관리", st["table_head"])],
        [P("Python 경로", st["table"]), P("사용자 PC 절대경로 고정", st["table"]), P("환경변수 또는 프로젝트 상대 venv 탐색", st["table"])],
        [P("데이터 경로", st["table"]), P("src/config.py에 Excel 절대경로", st["table"]), P("환경별 config 분리", st["table"])],
        [P("Git 범위", st["table"]), P("git add -A 후 main push", st["table"]), P("예약 작업이 소스와 산출물을 모두 올린다는 점을 명시", st["table"])],
        [P("원격 충돌", st["table"]), P("pull --rebase 충돌 시 rebase abort", st["table"]), P("로그 감시와 실패 알림 추가", st["table"])],
        [P("Scheduler 로그", st["table"]), P("마지막 로그 하나로 덮어쓰기", st["table"]), P("날짜별 보관 또는 rotation", st["table"])],
        [P("역할 명칭", st["table"]), P("Registry가 최종 기준", st["table"]), P("Causal Rank 65=production, Legacy S0=challenger 유지", st["table"])],
    ]
    story.append(table(cautions, [33 * mm, 66 * mm, 73 * mm], st))
    story.append(P(
        "예약 BAT는 Dashboard만 생략할 뿐 테스트, 두 백테스트, bundle 생성, commit, pull --rebase, push를 모두 수행한다.",
        st["callout"],
    ))
    story.append(PageBreak())
    return story


def extension_page(st):
    story = [P("7. 기능 추가 시 구현 경로", st["h1"])]
    rows = [
        [P("추가하려는 기능", st["table_head"]), P("수정 위치", st["table_head"]), P("반드시 함께 확인", st["table_head"])],
        [P("새 데이터 시트", st["table"]), P("src/data_loader.py", st["table"]), P("필수/선택 시트 구분, 날짜 정렬, tail ffill 품질", st["table"])],
        [P("새 피처", st["table"]), P("src/features/<module>.py + assembly.py", st["table"]), P("feature group, core whitelist, per-date 결측 처리", st["table"])],
        [P("새 타깃", st["table"]), P("src/target_engine.py", st["table"]), P("run_variant.py의 Phase 3 cache token 필드", st["table"])],
        [P("새 모델", st["table"]), P("src/model_trainer.py + variant YAML", st["table"]), P("causal split, prediction 의미, checkpoint 무효화", st["table"])],
        [P("새 신호 오버레이", st["table"]), P("src/backtest.py", st["table"]), P("pre-overlay prediction 보존, 적용 순서, ablation", st["table"])],
        [P("새 최적화 제약", st["table"]), P("portfolio_optimizer._build_mvo_constraints", st["table"]), P("최초 최적화와 projection이 같은 feasible region 사용", st["table"])],
        [P("새 운영 지표", st["table"]), P("export_operating_data.py", st["table"]), P("bundle validator와 Streamlit 소비 경로", st["table"])],
    ]
    story.append(table(rows, [40 * mm, 58 * mm, 74 * mm], st))
    story.append(P("권장 구현 원칙", st["h2"]))
    principles = [
        [P("1", st["center"]), P("새 기능은 PipelineConfig와 Variant YAML로 제어하고 기본 OFF에서 시작", st["table"])],
        [P("2", st["center"]), P("신호 변경과 포트폴리오 변경을 분리해 harvest-once / re-MVO-many 비교 가능하게 유지", st["table"])],
        [P("3", st["center"]), P("오버레이는 단독 효과와 leave-one-out 효과를 모두 측정", st["table"])],
        [P("4", st["center"]), P("Optimizer와 projection이 동일한 hard constraint를 사용하도록 구현", st["table"])],
        [P("5", st["center"]), P("새 운영 산출물은 source hash, 기준일, 유니버스, 리밸런싱 메타데이터까지 검증", st["table"])],
        [P("6", st["center"]), P("run_variant.py의 SAFE_FOR_CACHE_REUSE에 넣기 전 upstream 결과 불변 여부를 증명", st["table"])],
    ]
    story.append(table(principles, [15 * mm, 157 * mm], st, header=False, row_bgs={0: LIGHT_BLUE, 2: LIGHT_BLUE, 4: LIGHT_BLUE}))
    story.append(P("주요 참조 파일", st["h2"]))
    refs = (
        "src/config.py | src/data_loader.py | src/features/assembly.py | src/target_engine.py | "
        "src/model_trainer.py | src/backtest.py | src/portfolio_optimizer.py | run_variant.py | "
        "run_pictet_adoption.py | scripts/export_operating_data.py | scripts/validate_portfolio_bundles.py | "
        "streamlit_app.py | run_dashboard.bat | run_and_upload.bat | run_and_upload_scheduled.bat"
    )
    story.append(P(refs, st["small"]))
    story.append(P(
        "최종 요약: 모델 점수의 품질만큼 중요한 것은 execution timing, benchmark-relative risk budget, "
        "동일 제약 재투영, 운영 bundle 검증이다. 이 네 가지가 실제 운용 가능한 포트폴리오를 만든다.",
        st["callout"],
    ))
    return story


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.45)
    canvas.line(20 * mm, h - 14 * mm, w - 20 * mm, h - 14 * mm)
    canvas.setFont("Malgun", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, h - 10.5 * mm, "AI Portfolio Logic & BAT Process")
    canvas.drawRightString(w - 20 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def build() -> Path:
    register_fonts()
    st = styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=19 * mm, bottomMargin=16 * mm,
        title="AI Portfolio Logic & BAT Process",
        author="OpenAI Codex",
        subject="ai_port portfolio logic and Windows BAT orchestration",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=header_footer)])
    story = []
    story += cover_page(st)
    story += architecture_page(st)
    story += signal_page(st)
    story += optimizer_page(st)
    story += operating_page(st)
    story += dashboard_bat_page(st)
    story += upload_bat_page(st)
    story += extension_page(st)
    doc.build(story)
    return OUT


if __name__ == "__main__":
    print(build())
