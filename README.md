# S&P 500 10-year data pipeline

현재 및 과거 S&P 500 구성종목을 대상으로 최근 10년 OHLCV를 수집·정제하고,
생존자 편향을 보정한 뒤 클러스터링/백테스팅에 바로 쓸 수 있는 피처·리밸런싱
데이터셋을 만드는 파이프라인입니다. 클러스터링 알고리즘 자체와 백테스팅 실행은
아직 범위에 포함하지 않습니다 (다음 단계).

## 실행 순서

프로젝트 루트에서 의존성을 설치한 뒤 `notebooks/`의 노트북을 순서대로 실행합니다.

```powershell
python -m pip install -r requirements.txt
```

Tiingo 폴백까지 쓰려면 프로젝트 루트에 `.env` 파일을 만들고 `TIINGO_API_KEY=...`를
넣어둡니다 (`.env`는 git에서 제외됨). 키가 없어도 yfinance/Yahoo chart 수집은
정상 실행되며, 최종 실패 사유에 Tiingo 키 부재가 기록됩니다.

1. `01_collect_data.ipynb` — 현재 + 과거 구성종목(생존자 편향 보정 유니버스)의
   10년 OHLCV와 S&P500 지수를 API에서 한 번 수집
2. `02_preprocessing.ipynb` — 저장된 raw 파일만 읽어 전처리(반복 실행 가능)
3. `03_check_data.ipynb` — 저장된 결과를 검증하고 상태를 요약(반복 실행 가능)
4. `04_new_colums.ipynb` — 60/120일 파생 피처 생성, 클러스터링용 피처 선별,
   리밸런싱 시점 샘플링 및 시점별 자격 필터까지 적용(반복 실행 가능)

## 산출물

### `data/raw/`

- `sp500_universe.csv`: 수집 대상 종목 명단 (현재 구성종목 + 2016~2026 사이
  한 번이라도 구성종목이었던 과거 편출 종목 포함)
- `final_df_raw.parquet`: 수집 완료 원본 가격 패널
- `final_missing_df.csv`: 수집 폴백(yfinance→Yahoo chart→Tiingo) 후 끝까지
  실패한 종목의 raw 체크포인트
- `sp500_beta_df.parquet`: S&P500 지수(`^GSPC`) 자체의 일별 종가. 베타 계산용
  벤치마크이며 개별 종목 패널과 절대 섞이지 않음
- `cache/sp500_membership_history.csv`: 티커별 S&P500 편입일~편출일 이력 캐시
  (생존자 편향 보정의 근거 데이터)

### `data/processed/`

- `final_df.parquet`: 정제된 최종 OHLCV 데이터셋 (OHLC 논리 정합성 수정 완료)
- `coverage_df.csv`: 종목별 10년 coverage 및 품질 플래그 요약
- `final_missing_df.csv`: 수집·기간·전처리 후 최종 사용 불가능한 종목과 사유
- `final_60_df.parquet` / `final_120_df.parquet`: 60일/120일 기준 파생 피처
  8종(변동성, MDD, 하방변동성, 베타, 이격도, RSI, 모멘텀/수익률, CAGR) 포함
- `cluster_60df.parquet` / `cluster_120df.parquet`: 다중공선성 검증 후 확정한
  클러스터링용 피처 4개(`beta`, `volatility`, `return`, `rsi`)만 추린 버전 —
  아직 군집화 알고리즘을 돌리기 전 입력 데이터
- `rebalance_60df.parquet` / `rebalance_120df.parquet`: 60일/120일 간격
  리밸런싱 시점 스냅샷. 각 (종목, 날짜) 조합이 그 시점에 실제 S&P500
  구성종목이었는지까지 필터링 완료 — 백테스팅에 바로 사용 가능

대용량 parquet와 수집 캐시는 Git에서 제외됩니다.

## 생존자 편향(Survivorship Bias) 보정

`get_sp500_universe()`는 "오늘 기준" 구성종목만 반환하므로, 이것만으로 10년치를
분석하면 그 사이 상장폐지·인수합병으로 편출된 종목이 처음부터 존재하지 않는
셈이 되어 백테스팅 성과가 실제보다 좋게 나오는 착시가 생깁니다. 이를 보정하기
위해 다음 두 단계를 적용합니다.

1. **유니버스 확장** — `get_historical_sp500_universe(start, end)`가
   `fja05680/sp500` 저장소의 편입/편출 이력을 이용해, 분석 기간 중 한 번이라도
   구성종목이었던 모든 티커를 복원합니다. 현재 503개 → 확장 729개(226개 추가,
   그중 207개 실제 가격 데이터 복구 성공).
2. **리밸런싱 시점별 자격 필터** — `filter_by_membership(df)`가 각
   (종목, 날짜) 조합이 실제로 그 날짜에 S&P500 구성종목이었는지 편입/편출
   이력으로 재검증합니다. 단순히 "그 날짜에 가격 데이터가 있음"만으로는
   부족하기 때문입니다(예: 편출 후에도 계속 거래되는 종목, 최근 재편입된
   종목). `rebalance_60df` 기준 전체 조합의 22.4%가 이 필터로 제외됩니다.

## 모듈 역할

- `src/get_tickers.py`: 현재 구성종목 수집(`get_sp500_universe`), 과거 편입/
  편출 이력 기반 유니버스 확장(`get_historical_sp500_universe`), 리밸런싱
  시점별 자격 필터(`filter_by_membership`), 티커 정규화
- `src/collect_prices.py`: yfinance → Yahoo chart → Tiingo 3단계 폴백
  수집(`make_df`), S&P500 지수 자체 수집(`collect_sp500_index`)
- `src/preprocess.py`: 타입·정렬·중복·결측 처리, OHLC 논리 정합성 수정,
  기본 품질 플래그·coverage 처리
- `src/feature.py`: 종목별 60/120일 롤링 파생 피처 계산(`add_features`).
  각 행은 그 날짜 이전 데이터만 사용해 lookahead bias를 원천 차단
- `src/validate.py`: 최종 스키마와 universe/성공/실패/coverage 정합성 검증
