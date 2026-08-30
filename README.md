# S&P 500 Low Volatility 팩터 전략 — 데이터 파이프라인부터 백테스트까지

현재 및 과거 S&P 500 구성종목을 대상으로 최근 10년 OHLCV를 수집·정제하고,
생존자 편향을 보정한 뒤 **안정성(beta/volatility) 우선 클러스터링, 지도학습
기반 종목 선정, 재무지표(PER/PBR/ROE/부채비율/매출성장률/시가총액) 확장,
거래비용을 반영한 전체 백테스트**까지 전 과정을 다루는 파이프라인입니다.
데이터 수집부터 최종 전략 성과 검증까지 전 범위가 구현되어 있습니다.

## 실행 순서

프로젝트 루트에서 의존성을 설치한 뒤 `notebooks/`의 노트북을 순서대로 실행합니다.

```powershell
python -m pip install -r requirements.txt
```

Tiingo 폴백까지 쓰려면 프로젝트 루트에 `.env` 파일을 만들고 `TIINGO_API_KEY=...`를
넣어둡니다 (`.env`는 git에서 제외됨). 키가 없어도 yfinance/Yahoo chart 수집은
정상 실행되며, 최종 실패 사유에 Tiingo 키 부재가 기록됩니다.

1. `01_collect_data.ipynb` — 현재 + 과거 구성종목(생존자 편향 보정 유니버스)의
   10년 OHLCV와 S&P500 지수를 API에서 한 번 수집 (유일하게 네트워크를 쓰는 노트북)
2. `02_preprocessing.ipynb` — 저장된 raw 파일만 읽어 전처리, 데이터 오염 종목
   제외(PARA 등)까지 적용 (반복 실행 가능)
3. `03_check_data.ipynb` — 저장된 결과를 검증하고 상태를 요약 (반복 실행 가능)
4. `04_new_colums.ipynb` — 60/120일 파생 피처 생성(`beta`/`volatility`/`return`/`rsi`
   4개, 극단값 NaN 마스킹 적용), 리밸런싱 시점 샘플링 및 시점별 자격 필터까지 적용
5. `05_clustering.ipynb` — Ward 계층적 클러스터링으로 안정성 기준 2개 그룹으로
   분리(`n_clusters`는 실루엣 스코어로 검증해 2로 확정), 시각화 포함
6. `06_performance.ipynb` — 안정 그룹 vs 나머지 그룹의 성과 비교, 시장 국면(베타
   노출)과의 상관관계 분석, Sharpe ratio 정식 계산
7. `07_robustness.ipynb` — 30/60/120/252일 4개 리밸런싱 간격으로 확장해 결론의
   강건성 검증 (안정 그룹 승리 2/패배 2 — 시장 상황에 따라 갈림을 확인)
8. `08_ml_selection.ipynb` — 랜덤포레스트·로지스틱회귀·XGBoost로 "다음 기간
   상대적으로 잘 나갈 종목"을 예측하는 지도학습 파이프라인. 70/30 시간 분할 +
   연도별 walk-forward 검증(expanding window) 둘 다 수행
9. `09_per_experiment.ipynb` — point-in-time PER(주가수익비율)을 추가했을 때
   예측력이 개선되는지 5년 통제 실험으로 검증
10. `10_full_backtest.ipynb` — 저변동성 클러스터링 / ML 합의추천 / ML+PER 세
    전략을 시계열 복리 곱선으로 이어붙여 S&P500 인덱스와 비교, 거래비용 반영
11. `11_fundamentals_expansion.ipynb` — PBR/ROE/부채비율/매출성장률/log
    시가총액까지 확장해 6-팩터(Risk/Momentum/Growth/Valuation/Quality/Size)
    구조로 재구성, 팩터 레벨 피처 중요도 분석

## 핵심 발견

- **안정성(beta/volatility)이 낮은 종목군의 초과성과는 시장 국면에 좌우된다** —
  안정-나머지 수익률 격차가 같은 기간 S&P500 수익률과 뚜렷한 음(-)의
  상관관계(60일 -0.70, 120일 -0.53)를 보인다. 강세장에서는 고베타 종목이
  앞서고, 약세·횡보장에서는 안정 그룹이 방어력을 갖는 전형적인 저변동성
  팩터 패턴이다.
- **08번(10년, 가격 기반 지표만) 실험에서는 개별 종목의 상대적 성과를 예측하지
  못했다** — 지도학습 모델 3종 모두 정확도·AUC가 약 0.49~0.50 수준(동전 던지기
  수준)이었고, 연도별 walk-forward 검증(2021~2026, 6개 fold)에서도 모든 해에
  걸쳐 일관되게 예측력을 확인하지 못했다. (11번처럼 재무지표까지 포함한 더 짧은
  기간·다른 표본에서는 baseline AUC가 0.52대로 다르게 나온다 — 아래 참고.)
- **재무지표를 추가해도 실전 포트폴리오 성과는 개선되지 않는다** — PER 하나만
  추가했을 때는 작지만 일관된 개선이 있었지만, PBR/ROE/부채비율/매출성장률/
  시가총액까지 6개 재무지표로 확장하자 예측 정확도는 미세하게만 오르고 합의
  추천 포트폴리오의 위험대비수익은 오히려 단조 감소했다(0.187→0.166→0.160).
- **팩터 레벨로 보면 Risk(beta·volatility·부채비율)가 압도적으로 가장 중요한
  축이다** — 지도학습 피처 중요도에서 Risk 팩터 하나가 나머지 5개 팩터를
  합친 것보다 크다. 비지도 클러스터링과 지도학습에서 모두 Risk 관련 지표가
  중요한 축으로 나타나며, "안정성 우선"이라는 초기 가설과 일관된 결과를
  확인했다. (다만 두 방법론이 같은 데이터와 일부 동일한 가격 기반 피처를
  공유하므로, "완전히 독립적인 증거"라고까지 말하기는 어렵다.)
- **세 전략(저변동성 클러스터링 / ML 합의추천 / ML+PER) 모두 거래비용 반영
  후에도 단순 S&P500 인덱스 투자를 이기지 못했다.** ML 기반 전략의 회전율
  (56~61%)이 클러스터링 기반(24%)보다 훨씬 높아 거래비용에 더 취약하다는
  점도 발견했다.

## 데이터 품질 — 발견하고 수정한 두 가지 오염

- **극단값 오염**: 미조정 분할·상장폐지 후 티커 재사용 등으로 특정 종목의
  단일 거래일 가격이 비정상적으로 튀는 경우(예: SIVB, STI, CHK), 롤링 윈도우
  계산 특성상 그 한 행이 window 전체를 오염시켜 beta/volatility가 비정상
  범위(최대 400대)까지 폭주했다. `find_price_anomalies()`로 원본을 먼저
  스캔하고, 일간수익률·기간수익률·beta 각각에 NaN 마스킹 가드를 추가해
  해결했다.
- **PARA 종목 가격 스케일 오염**: 특정 구간(2021~2023)의 Close 가격이 실제
  주가보다 약 1000배 큰 값으로 저장되어 있었다(채권 가격 시스템의 전형적
  특징 — 낮은 변동성, 액면가 근처 움직임). 단일 거래일 극단치가 아니라 여러
  해에 걸친 완만한 오염이라 기존 가드로는 못 잡아서, 종목 자체를 제외했다.

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
- `cache/eps_history.csv`, `cache/income_statement_history.csv`,
  `cache/balance_sheet_history.csv`: yfinance 연간 재무제표 캐시(EPS/순이익/
  매출, 자기자본/부채/발행주식수) — 09/11번 노트북의 재무지표 실험용

### `data/processed/`

- `final_df.parquet`: 정제된 최종 OHLCV 데이터셋 (OHLC 논리 정합성 수정,
  데이터 오염 종목 제외 완료)
- `coverage_df.csv`: 종목별 10년 coverage 및 품질 플래그 요약
- `final_missing_df.csv`: 수집·기간·전처리 후 최종 사용 불가능한 종목과 사유
- `final_60_df.parquet` / `final_120_df.parquet`: 60일/120일 기준 파생 피처
  4종(`beta`, `volatility`, `return`, `rsi`) — 다중공선성 검토 후 8개에서
  4개로 확정한 버전. 극단값 NaN 마스킹 적용됨
- `cluster_60df.parquet` / `cluster_120df.parquet`, `clustered_*df.parquet`:
  클러스터링 입력/출력 (30/60/120/252일 4개 간격 전부 생성)
- `rebalance_*df.parquet` (30/60/120/252일): 리밸런싱 시점 스냅샷. 각
  (종목, 날짜) 조합이 그 시점에 실제 S&P500 구성종목이었는지까지 필터링
  완료 — 백테스팅에 바로 사용 가능

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
  기본 품질 플래그·coverage 처리(`build_ml_dataset`)
- `src/feature.py`: 종목별 30/60/120/252일 롤링 파생 피처 계산(`add_features`).
  각 행은 그 날짜 이전 데이터만 사용해 lookahead bias를 원천 차단.
  극단값 NaN 마스킹(`find_price_anomalies`), 옵션형 고유변동성 계산 포함
- `src/validate.py`: 최종 스키마와 universe/성공/실패/coverage 정합성 검증
- `src/cluster.py`: 리밸런싱 시점별 Ward 계층적 클러스터링으로 안정성 그룹
  분리(`cluster_snapshot`, `cluster_all_snapshots`)
- `src/performance.py`: 클러스터별 성과 집계·위험대비수익·Sharpe ratio 계산
  (`add_forward_returns`, `summarize_by_group`, `sharpe_ratio_stats`)
- `src/backtest.py`: 시계열 복리 누적수익 곡선, 최대낙폭(MDD), 연환산 Sharpe,
  회전율 기반 거래비용 반영(`equity_curve`, `max_drawdown`,
  `apply_transaction_costs`)
- `src/ml_selection.py`: 상대수익률 기반 라벨링, 70/30 시간분할 및 연도별
  walk-forward 분할, 모델 3종(RF/로지스틱/XGBoost) 학습·예측, 6-팩터 매핑
  (`FACTOR_GROUPS`, `metric_to_factor`)
- `src/fundamentals.py`: yfinance 연간 재무제표에서 point-in-time PER/PBR/
  ROE/부채비율/매출성장률/log시가총액 계산 (`REPORTING_LAG_DAYS`로 보고
  지연을 반영해 lookahead bias 차단)
