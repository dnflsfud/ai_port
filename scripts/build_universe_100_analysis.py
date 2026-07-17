"""Build and execute the reproducible 65-to-100 global universe analysis.

The script writes a minimal nbformat-v4 notebook, executes each Python cell in
order, and persists reviewed result tables to JSON and SQLite for the report.
"""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "universe_100_recommendation"
NOTEBOOK_PATH = OUTPUT_DIR / "universe_100_analysis.ipynb"


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    markdown_cell(
        """# 65종목 유니버스의 글로벌 100종목 확장 분석

## tl;dr

- 현재 유니버스는 65종목이며 경제적 본거지 기준 미국 60개(92.3%), 한국 2개, 대만·영국·태국 각 1개다.
- 마지막 운영 포트폴리오(2026-05-22)의 기술주 비중은 68.5%로, 종목 수 기준 기술주 비중 36.9%보다 훨씬 높다.
- 신규 35종목은 미국 9개, 영국 8개, 유럽(영국 제외) 17개, 일본 1개로 배분한다. 최종 100종목의 지역 구성은 미국 69%, 영국 9%, 유럽(영국 제외) 17%, 아시아 5%다.
- 키옥시아(285A JP)와 샌디스크(SNDK US)는 필수 포함하되, 짧은 상장 이력 때문에 최소 이력 게이트를 적용한다.
"""
    ),
    markdown_cell(
        """## Context & Methods

### Key Assumptions

- 섹터 분류는 원천 파일 `Universe_Meta`의 체계를 그대로 사용한다.
- 국가와 지역은 거래소가 아니라 발행기업의 경제적 본거지를 기준으로 분류한다. 따라서 TSM US는 대만으로 분류한다.
- 현재 비중은 종목 수 기준과 마지막 운영 산출물의 포트폴리오·벤치마크 비중을 분리한다.
- 신규 종목은 기대수익률 순위가 아니라 공급망 커버리지, 대형주 유동성, 장기 데이터 가용성, 섹터 및 지역 분산을 기준으로 제안한다.
- 키옥시아와 샌디스크는 사용자 지정 필수 편입 종목이다.
"""
    ),
    code_cell(
        """from pathlib import Path
import json
import pandas as pd

ROOT = Path(r'C:\\Users\\westl\\PycharmProjects\\pythonProject\\venv_vf_new\\machine\\re_study\\c2\\ai_port')
SOURCE_XLSX = Path(r'C:\\Users\\westl\\PycharmProjects\\pythonProject\\venv_vf_new\\machine\\re_study\\ai_signal_data.xlsx')
OPERATIONS_JSON = ROOT / 'outputs' / 'operating' / 'operations.json'
PORTFOLIO_JSON = ROOT / 'outputs' / 'operating' / 'portfolio.json'
RESULTS_JSON = ROOT / 'outputs' / 'universe_100_recommendation' / 'universe_100_results.json'
SQLITE_PATH = ROOT / 'outputs' / 'universe_100_recommendation' / 'universe_100_analysis.sqlite'

candidates = [
    {'ticker':'285A JP','name':'Kioxia Holdings','sector':'Technology','subindustry':'NAND flash / SSD','country':'Japan','region':'Asia','priority':'Required','rationale':'NAND 공급망 직접 노출; 사용자 지정 필수 편입','listing_date':'2024-12-18','history_gate':True},
    {'ticker':'SNDK US','name':'Sandisk','sector':'Technology','subindustry':'Flash memory / SSD','country':'United States','region':'United States','priority':'Required','rationale':'클라이언트·데이터센터 플래시 메모리 커버리지; 사용자 지정 필수 편입','listing_date':'2025-02-24','history_gate':True},
    {'ticker':'ASML NA','name':'ASML Holding','sector':'Technology','subindustry':'Lithography equipment','country':'Netherlands','region':'Europe ex-UK','priority':'Core','rationale':'첨단 반도체 노광 장비의 핵심 공급망','listing_date':None,'history_gate':False},
    {'ticker':'SAP GR','name':'SAP','sector':'Technology','subindustry':'Enterprise software','country':'Germany','region':'Europe ex-UK','priority':'Core','rationale':'유럽 대표 엔터프라이즈 소프트웨어 플랫폼','listing_date':None,'history_gate':False},
    {'ticker':'STM FP','name':'STMicroelectronics','sector':'Technology','subindustry':'Analog / automotive semiconductors','country':'Netherlands','region':'Europe ex-UK','priority':'Core','rationale':'자동차·산업용 반도체로 AI 일변도 완화','listing_date':None,'history_gate':False},
    {'ticker':'KLAC US','name':'KLA','sector':'Technology','subindustry':'Process control equipment','country':'United States','region':'United States','priority':'Core','rationale':'검사·계측 장비로 전공정 장비 노출 보완','listing_date':None,'history_gate':False},
    {'ticker':'QCOM US','name':'Qualcomm','sector':'Technology','subindustry':'Wireless / edge semiconductors','country':'United States','region':'United States','priority':'Core','rationale':'모바일·엣지 AI 반도체 노출','listing_date':None,'history_gate':False},
    {'ticker':'MRVL US','name':'Marvell Technology','sector':'Technology','subindustry':'Data-center networking chips','country':'United States','region':'United States','priority':'Core','rationale':'데이터센터 인터커넥트·맞춤형 실리콘 커버리지','listing_date':None,'history_gate':False},
    {'ticker':'CDNS US','name':'Cadence Design Systems','sector':'Technology','subindustry':'EDA software','country':'United States','region':'United States','priority':'Core','rationale':'반도체 설계 소프트웨어 공급망','listing_date':None,'history_gate':False},
    {'ticker':'DELL US','name':'Dell Technologies','sector':'Technology','subindustry':'Servers / enterprise hardware','country':'United States','region':'United States','priority':'Core','rationale':'AI 서버·기업 하드웨어 수요 커버리지','listing_date':None,'history_gate':False},
    {'ticker':'SIE GR','name':'Siemens','sector':'Industrials','subindustry':'Industrial automation','country':'Germany','region':'Europe ex-UK','priority':'Diversifier','rationale':'산업 자동화·전력화 노출','listing_date':None,'history_gate':False},
    {'ticker':'SU FP','name':'Schneider Electric','sector':'Industrials','subindustry':'Electrification / automation','country':'France','region':'Europe ex-UK','priority':'Diversifier','rationale':'데이터센터 전력관리와 자동화 노출','listing_date':None,'history_gate':False},
    {'ticker':'RR/ LN','name':'Rolls-Royce Holdings','sector':'Industrials','subindustry':'Aerospace engines','country':'United Kingdom','region':'United Kingdom','priority':'Diversifier','rationale':'민항 엔진·장기 서비스 매출 노출','listing_date':None,'history_gate':False},
    {'ticker':'PWR US','name':'Quanta Services','sector':'Industrials','subindustry':'Grid infrastructure','country':'United States','region':'United States','priority':'Diversifier','rationale':'전력망·데이터센터 전력 인프라 수혜','listing_date':None,'history_gate':False},
    {'ticker':'HSBA LN','name':'HSBC Holdings','sector':'Financials','subindustry':'Global banking','country':'United Kingdom','region':'United Kingdom','priority':'Diversifier','rationale':'아시아 비중이 큰 글로벌 은행 노출','listing_date':None,'history_gate':False},
    {'ticker':'ALV GR','name':'Allianz','sector':'Financials','subindustry':'Insurance / asset management','country':'Germany','region':'Europe ex-UK','priority':'Diversifier','rationale':'보험·자산운용 복합 금융 노출','listing_date':None,'history_gate':False},
    {'ticker':'UBSG SW','name':'UBS Group','sector':'Financials','subindustry':'Wealth management / banking','country':'Switzerland','region':'Europe ex-UK','priority':'Diversifier','rationale':'글로벌 자산관리 중심 금융 플랫폼','listing_date':None,'history_gate':False},
    {'ticker':'LSEG LN','name':'London Stock Exchange Group','sector':'Financials','subindustry':'Market infrastructure / data','country':'United Kingdom','region':'United Kingdom','priority':'Diversifier','rationale':'거래소·금융데이터 인프라 노출','listing_date':None,'history_gate':False},
    {'ticker':'NOVOB DC','name':'Novo Nordisk','sector':'Healthcare','subindustry':'Diabetes / obesity pharmaceuticals','country':'Denmark','region':'Europe ex-UK','priority':'Diversifier','rationale':'비만·당뇨 치료제의 글로벌 성장 노출','listing_date':None,'history_gate':False},
    {'ticker':'AZN LN','name':'AstraZeneca','sector':'Healthcare','subindustry':'Pharmaceuticals','country':'United Kingdom','region':'United Kingdom','priority':'Diversifier','rationale':'종양·희귀질환 중심 대형 제약 노출','listing_date':None,'history_gate':False},
    {'ticker':'ROG SW','name':'Roche Holding','sector':'Healthcare','subindustry':'Pharmaceuticals / diagnostics','country':'Switzerland','region':'Europe ex-UK','priority':'Diversifier','rationale':'제약과 진단의 결합 노출','listing_date':None,'history_gate':False},
    {'ticker':'NOVN SW','name':'Novartis','sector':'Healthcare','subindustry':'Innovative medicines','country':'Switzerland','region':'Europe ex-UK','priority':'Diversifier','rationale':'유럽 대형 혁신신약 포트폴리오','listing_date':None,'history_gate':False},
    {'ticker':'TMO US','name':'Thermo Fisher Scientific','sector':'Healthcare','subindustry':'Life-science tools','country':'United States','region':'United States','priority':'Diversifier','rationale':'생명과학 도구·서비스 커버리지','listing_date':None,'history_gate':False},
    {'ticker':'MC FP','name':'LVMH','sector':'Consumer Discretionary','subindustry':'Luxury goods','country':'France','region':'Europe ex-UK','priority':'Diversifier','rationale':'글로벌 럭셔리 소비와 브랜드 파워 노출','listing_date':None,'history_gate':False},
    {'ticker':'RACE IM','name':'Ferrari','sector':'Consumer Discretionary','subindustry':'Luxury automobiles','country':'Italy','region':'Europe ex-UK','priority':'Diversifier','rationale':'희소성이 높은 럭셔리 자동차 노출','listing_date':None,'history_gate':False},
    {'ticker':'BKNG US','name':'Booking Holdings','sector':'Consumer Discretionary','subindustry':'Online travel','country':'United States','region':'United States','priority':'Diversifier','rationale':'글로벌 여행 플랫폼 노출','listing_date':None,'history_gate':False},
    {'ticker':'SHEL LN','name':'Shell','sector':'Energy','subindustry':'Integrated energy','country':'United Kingdom','region':'United Kingdom','priority':'Diversifier','rationale':'통합 에너지와 LNG 노출','listing_date':None,'history_gate':False},
    {'ticker':'TTE FP','name':'TotalEnergies','sector':'Energy','subindustry':'Integrated energy','country':'France','region':'Europe ex-UK','priority':'Diversifier','rationale':'석유·가스와 전력 전환 포트폴리오','listing_date':None,'history_gate':False},
    {'ticker':'ULVR LN','name':'Unilever','sector':'Consumer Staples','subindustry':'Household / personal products','country':'United Kingdom','region':'United Kingdom','priority':'Diversifier','rationale':'글로벌 생활소비재의 방어적 현금흐름','listing_date':None,'history_gate':False},
    {'ticker':'NESN SW','name':'Nestle','sector':'Consumer Staples','subindustry':'Packaged food / beverages','country':'Switzerland','region':'Europe ex-UK','priority':'Diversifier','rationale':'글로벌 식품·음료 브랜드 노출','listing_date':None,'history_gate':False},
    {'ticker':'DTE GR','name':'Deutsche Telekom','sector':'Communication Services','subindustry':'Telecom','country':'Germany','region':'Europe ex-UK','priority':'Diversifier','rationale':'유럽 통신과 미국 무선 자회사 노출','listing_date':None,'history_gate':False},
    {'ticker':'VOD LN','name':'Vodafone Group','sector':'Communication Services','subindustry':'Telecom','country':'United Kingdom','region':'United Kingdom','priority':'Diversifier','rationale':'유럽·아프리카 통신 노출','listing_date':None,'history_gate':False},
    {'ticker':'VNA GR','name':'Vonovia','sector':'Real Estate','subindustry':'Residential real estate','country':'Germany','region':'Europe ex-UK','priority':'Diversifier','rationale':'유럽 주거 부동산과 금리 민감도 노출','listing_date':None,'history_gate':False},
    {'ticker':'RIO LN','name':'Rio Tinto','sector':'Materials','subindustry':'Diversified mining','country':'United Kingdom','region':'United Kingdom','priority':'Diversifier','rationale':'철광석·알루미늄·구리의 글로벌 광산 노출','listing_date':None,'history_gate':False},
    {'ticker':'IBE SM','name':'Iberdrola','sector':'Utilities','subindustry':'Electric utility / renewables','country':'Spain','region':'Europe ex-UK','priority':'Diversifier','rationale':'전력망과 재생에너지 중심 유틸리티 노출','listing_date':None,'history_gate':False},
]

print(f'Candidates defined: {len(candidates)}')
"""
    ),
    markdown_cell(
        """## Data

원천 엑셀에서 현재 종목·섹터 메타데이터와 가격 데이터 범위를 읽고, 운영 JSON에서 마지막 포트폴리오·벤치마크 섹터 비중을 읽는다.
"""
    ),
    code_cell(
        """meta = pd.read_excel(SOURCE_XLSX, sheet_name='Universe_Meta')
prices_date = pd.read_excel(SOURCE_XLSX, sheet_name='PX_LAST', usecols=['date'])['date']
operations = json.loads(OPERATIONS_JSON.read_text(encoding='utf-8'))
portfolio_meta = json.loads(PORTFOLIO_JSON.read_text(encoding='utf-8'))

print(f'Universe rows: {len(meta)}')
print(f'Universe status values: {meta["Status"].value_counts(dropna=False).to_dict()}')
print(f'Price data range: {prices_date.min().date()} to {prices_date.max().date()}')
print(f'Operating portfolio as of: {operations["as_of"]}')
print(f'Portfolio export data as of: {portfolio_meta["data_as_of"]}')
"""
    ),
    markdown_cell("""## Results

### 1. 현재 섹터 구성과 마지막 운영 비중
"""),
    code_cell(
        """current_counts = meta.groupby('Sector').size().rename('current_names').reset_index()
current_counts['current_name_share'] = current_counts['current_names'] / len(meta)

exposure = pd.DataFrame(operations['sector_exposure']).T.rename_axis('Sector').reset_index()
current = current_counts.merge(exposure, on='Sector', how='left')
current['active'] = current['portfolio'] - current['benchmark']
current = current.sort_values(['current_names','Sector'], ascending=[False,True])

print(current.to_string(index=False, formatters={
    'current_name_share': lambda x: f'{x:.1%}',
    'portfolio': lambda x: f'{x:.1%}',
    'benchmark': lambda x: f'{x:.1%}',
    'active': lambda x: f'{x:+.1%}',
}))
"""
    ),
    markdown_cell("""### 2. 신규 35종목과 최종 100종목 섹터 구성
"""),
    code_cell(
        """candidate_df = pd.DataFrame(candidates)
add_counts = candidate_df.groupby('sector').size().rename('add_names')
proposed = current_counts.set_index('Sector').join(add_counts, how='outer').fillna(0)
proposed[['current_names','add_names']] = proposed[['current_names','add_names']].astype(int)
proposed['final_names'] = proposed['current_names'] + proposed['add_names']
proposed['final_name_share'] = proposed['final_names'] / proposed['final_names'].sum()
proposed = proposed.reset_index().sort_values(['final_names','Sector'], ascending=[False,True])

print(proposed.to_string(index=False, formatters={'final_name_share': lambda x: f'{x:.1%}'}))
print('\\nRecommended additions:')
for sector, group in candidate_df.groupby('sector', sort=False):
    print(f'- {sector} ({len(group)}): ' + ', '.join(group['ticker']))
"""
    ),
    markdown_cell("""### 3. 국가·지역 편중 완화
"""),
    code_cell(
        """def current_country(ticker):
    if ticker.startswith(('000660 KS', '005930 KS')):
        return 'South Korea'
    if ticker.startswith('TSM US'):
        return 'Taiwan'
    if ticker.startswith('LIN US'):
        return 'United Kingdom'
    if ticker.startswith('FN US'):
        return 'Thailand'
    return 'United States'

def country_region(country):
    if country == 'United States':
        return 'United States'
    if country == 'United Kingdom':
        return 'United Kingdom'
    if country in {'South Korea', 'Taiwan', 'Japan', 'Thailand'}:
        return 'Asia'
    return 'Europe ex-UK'

current_geo = meta[['Ticker']].copy()
current_geo['country'] = current_geo['Ticker'].map(current_country)
current_geo['region'] = current_geo['country'].map(country_region)

current_country_counts = current_geo.groupby('country').size().rename('current_names')
add_country_counts = candidate_df.groupby('country').size().rename('add_names')
proposed_country = pd.concat([current_country_counts, add_country_counts], axis=1).fillna(0).astype(int)
proposed_country['final_names'] = proposed_country['current_names'] + proposed_country['add_names']
proposed_country['final_share'] = proposed_country['final_names'] / proposed_country['final_names'].sum()
proposed_country = proposed_country.reset_index().sort_values(['final_names','country'], ascending=[False,True])

region_current = current_geo.groupby('region').size().rename('current_names')
region_add = candidate_df.groupby('region').size().rename('add_names')
region_order = ['United States', 'United Kingdom', 'Europe ex-UK', 'Asia']
region_mix = pd.concat([region_current, region_add], axis=1).fillna(0).astype(int).reindex(region_order, fill_value=0)
region_mix['final_names'] = region_mix['current_names'] + region_mix['add_names']
region_mix['current_share'] = region_mix['current_names'] / len(meta)
region_mix['final_share'] = region_mix['final_names'] / 100
region_mix = region_mix.reset_index().rename(columns={'index':'region'})

print('Current country counts:')
print(current_geo.groupby('country').size().sort_values(ascending=False).to_string())
print('\\nProposed country counts:')
print(proposed_country.to_string(index=False, formatters={'final_share': lambda x: f'{x:.1%}'}))
print('\\nRegion mix:')
print(region_mix.to_string(index=False, formatters={
    'current_share': lambda x: f'{x:.1%}',
    'final_share': lambda x: f'{x:.1%}',
}))
"""
    ),
    markdown_cell("""### 4. 데이터 품질 및 적용 게이트
"""),
    code_cell(
        """existing_short = set(meta['Ticker'].str.split().str[0])
candidate_short = candidate_df['ticker'].str.split().str[0]

checks = {
    'current_universe_is_65': len(meta) == 65,
    'all_current_status_available': bool(meta['Status'].eq('Available').all()),
    'current_sector_complete': int(meta['Sector'].isna().sum()) == 0,
    'candidate_count_is_35': len(candidate_df) == 35,
    'candidate_tickers_unique': candidate_df['ticker'].is_unique,
    'no_overlap_with_current_universe': not bool(set(candidate_short) & existing_short),
    'required_kioxia_present': '285A JP' in set(candidate_df['ticker']),
    'required_sandisk_present': 'SNDK US' in set(candidate_df['ticker']),
    'final_universe_is_100': int(proposed['final_names'].sum()) == 100,
    'final_country_count_is_100': int(proposed_country['final_names'].sum()) == 100,
    'final_us_is_69': int(region_mix.loc[region_mix['region'].eq('United States'), 'final_names'].iloc[0]) == 69,
    'final_uk_is_9': int(region_mix.loc[region_mix['region'].eq('United Kingdom'), 'final_names'].iloc[0]) == 9,
    'final_europe_ex_uk_is_17': int(region_mix.loc[region_mix['region'].eq('Europe ex-UK'), 'final_names'].iloc[0]) == 17,
    'final_asia_is_5': int(region_mix.loc[region_mix['region'].eq('Asia'), 'final_names'].iloc[0]) == 5,
    'current_name_shares_sum_to_one': abs(float(current_counts['current_name_share'].sum()) - 1.0) < 1e-12,
    'portfolio_sector_weights_sum_to_one': abs(float(exposure['portfolio'].sum()) - 1.0) < 0.001,
    'benchmark_sector_weights_sum_to_one': abs(float(exposure['benchmark'].sum()) - 1.0) < 0.001,
}

history_gates = candidate_df[candidate_df['history_gate']][
    ['ticker','name','listing_date','rationale']
].to_dict(orient='records')

print(pd.Series(checks, name='passed').to_string())
print('\\nHistory-gated additions:')
print(candidate_df[candidate_df['history_gate']][['ticker','name','listing_date']].to_string(index=False))

assert all(checks.values()), checks
"""
    ),
    code_cell(
        """import sqlite3

current_sector_long = pd.DataFrame([
    {
        'sector': row['Sector'], 'metric': metric, 'value': row[field],
        'current_names': row['current_names'], 'current_name_share': row['current_name_share'],
        'portfolio': row['portfolio'], 'benchmark': row['benchmark'], 'active': row['active'],
        'portfolio_as_of': operations['as_of'],
    }
    for _, row in current.iterrows()
    for metric, field in [('Name share', 'current_name_share'), ('Portfolio weight', 'portfolio'), ('Benchmark weight', 'benchmark')]
])

proposed_sector_long = pd.DataFrame([
    {
        'sector': row['Sector'], 'metric': metric, 'value': row[field],
        'current_names': row['current_names'], 'add_names': row['add_names'],
        'final_names': row['final_names'], 'current_name_share': row['current_name_share'],
        'final_name_share': row['final_name_share'],
    }
    for _, row in proposed.iterrows()
    for metric, field in [('Current', 'current_names'), ('Proposed', 'final_names')]
])

region_mix_long = pd.DataFrame([
    {
        'region': row['region'], 'metric': metric, 'value': row[field],
        'current_names': row['current_names'], 'add_names': row['add_names'],
        'final_names': row['final_names'], 'current_share': row['current_share'],
        'final_share': row['final_share'],
    }
    for _, row in region_mix.iterrows()
    for metric, field in [('Current', 'current_share'), ('Proposed', 'final_share')]
])

summary = pd.DataFrame([{
    'current_names': len(meta),
    'new_names': len(candidate_df),
    'final_names': int(proposed['final_names'].sum()),
    'tech_name_share': float(current.loc[current['Sector'].eq('Technology'), 'current_name_share'].iloc[0]),
    'tech_portfolio': float(current.loc[current['Sector'].eq('Technology'), 'portfolio'].iloc[0]),
    'tech_benchmark': float(current.loc[current['Sector'].eq('Technology'), 'benchmark'].iloc[0]),
    'final_tech_share': float(proposed.loc[proposed['Sector'].eq('Technology'), 'final_name_share'].iloc[0]),
    'current_us_share': float(region_mix.loc[region_mix['region'].eq('United States'), 'current_share'].iloc[0]),
    'final_us_share': float(region_mix.loc[region_mix['region'].eq('United States'), 'final_share'].iloc[0]),
    'final_uk_share': float(region_mix.loc[region_mix['region'].eq('United Kingdom'), 'final_share'].iloc[0]),
    'final_europe_ex_uk_share': float(region_mix.loc[region_mix['region'].eq('Europe ex-UK'), 'final_share'].iloc[0]),
    'final_asia_share': float(region_mix.loc[region_mix['region'].eq('Asia'), 'final_share'].iloc[0]),
}])
summary['tech_share_change'] = summary['final_tech_share'] - summary['tech_name_share']
summary['current_non_us_share'] = 1 - summary['current_us_share']
summary['final_non_us_share'] = 1 - summary['final_us_share']

candidate_sql = candidate_df.copy()
candidate_sql.insert(0, 'selection_order', range(1, len(candidate_sql) + 1))
candidate_sql['history_gate_label'] = candidate_sql['history_gate'].map({True: 'Required', False: 'Standard'})

with sqlite3.connect(SQLITE_PATH) as connection:
    summary.to_sql('summary', connection, if_exists='replace', index=False)
    current_sector_long.to_sql('current_sector_long', connection, if_exists='replace', index=False)
    proposed_sector_long.to_sql('proposed_sector_long', connection, if_exists='replace', index=False)
    proposed_country.to_sql('proposed_country', connection, if_exists='replace', index=False)
    region_mix_long.to_sql('region_mix_long', connection, if_exists='replace', index=False)
    candidate_sql.to_sql('candidates', connection, if_exists='replace', index=False)

result = {
    'generated_at': pd.Timestamp.now(tz='Asia/Seoul').isoformat(),
    'source': {
        'workbook': SOURCE_XLSX.name,
        'price_data_as_of': str(prices_date.max().date()),
        'portfolio_as_of': operations['as_of'],
        'portfolio_export_data_as_of': portfolio_meta['data_as_of'],
        'sqlite_snapshot': SQLITE_PATH.name,
    },
    'summary': json.loads(summary.to_json(orient='records'))[0],
    'current_sector': current.to_dict(orient='records'),
    'proposed_sector': proposed.to_dict(orient='records'),
    'proposed_country': proposed_country.to_dict(orient='records'),
    'region_mix': region_mix.to_dict(orient='records'),
    'candidates': json.loads(candidate_df.to_json(orient='records', force_ascii=False)),
    'checks': checks,
    'history_gates': history_gates,
}
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote {RESULTS_JSON}')
print(f'Wrote {SQLITE_PATH}')
"""
    ),
    markdown_cell(
        """## Takeaways

1. **현재 유니버스의 가장 큰 구조적 문제는 기술주뿐 아니라 미국 편중이다.** 기술주는 65개 중 24개(36.9%)지만 운영 포트폴리오 비중은 68.5%이고, 미국 기업은 60개(92.3%)다.
2. **35개 추가안은 섹터 분산과 지역 분산을 동시에 개선한다.** 최종 섹터는 기술 34%, 산업재 13%, 금융 12%, 헬스케어 11%이며, 지역은 미국 69%, 영국 9%, 유럽(영국 제외) 17%, 아시아 5%다.
3. **키옥시아와 샌디스크는 구조적으로 유용하지만 즉시 완전 편입은 위험하다.** 둘 다 짧은 상장 이력 때문에 상장 후 최소 이력, 결측률, 거래일 정합성 게이트가 필요하다.
4. **글로벌 확장에는 모델 외 운영 보강이 필수다.** 현지 통화 가격의 기준통화 환산, 국가별 휴장일·시간대 정렬, 배당·기업행위 조정, 현지주식과 ADR 중복 방지를 검증해야 한다.
"""
    ),
]


def execute_notebook(cells: list[dict]) -> tuple[list[dict], str | None]:
    namespace: dict = {"__name__": "__main__"}
    execution_count = 0
    failure = None
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        source = "".join(cell["source"])
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(source, f"<cell-{execution_count}>", "exec"), namespace)
            text = stdout.getvalue()
            if text:
                cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": text.splitlines(keepends=True)}]
        except Exception:
            failure = traceback.format_exc()
            text = stdout.getvalue()
            outputs = []
            if text:
                outputs.append({"name": "stdout", "output_type": "stream", "text": text.splitlines(keepends=True)})
            outputs.append({
                "ename": "ExecutionError",
                "evalue": failure.splitlines()[-1],
                "output_type": "error",
                "traceback": failure.splitlines(),
            })
            cell["outputs"] = outputs
            break
    return cells, failure


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    executed_cells, failure = execute_notebook(CELLS)
    notebook = {
        "cells": executed_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    reloaded = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert reloaded["nbformat"] == 4
    assert isinstance(reloaded["cells"], list) and reloaded["cells"]
    assert all(cell.get("cell_type") in {"markdown", "code"} for cell in reloaded["cells"])
    if failure:
        raise RuntimeError(f"Notebook execution failed:\n{failure}")
    print(f"Executed notebook written to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
