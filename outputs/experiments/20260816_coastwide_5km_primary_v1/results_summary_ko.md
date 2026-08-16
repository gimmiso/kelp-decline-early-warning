# 저널 보완 실험 결과

> 이 분석은 기존 결과를 확인한 뒤 수행한 reviewer robustness 분석이다. 사전등록 또는 독립 확증으로 표현하지 않는다.

## 1. 검증 프로토콜과 집계 방식

- 현재 상태 모델의 random-CV pooled AP lift는 0.164, expanding-window pooled AP lift는 0.160였다.
- 같은 모델의 random-CV macro within-year AP lift는 0.153, expanding-window 값은 0.153였다.
- random-CV는 미래 연도와 동일 셀의 다른 연도를 학습에 허용하는 의도적 비운영 비교이므로 최종 성능으로 사용하지 않는다.

## 2. 모델 계열 강건성

- Logistic: trajectory_minus_current +0.038 (CI +0.020–+0.057); oisst_minus_trajectory -0.017 (CI -0.038–+0.004); cuti_beuti_minus_oisst -0.009 (CI -0.024–+0.006).
- Random forest: trajectory_minus_current +0.042 (CI +0.027–+0.057); oisst_minus_trajectory +0.002 (CI -0.010–+0.015); cuti_beuti_minus_oisst -0.002 (CI -0.024–+0.016).
- XGBoost: trajectory_minus_current +0.033 (CI +0.020–+0.048); oisst_minus_trajectory +0.015 (CI -0.007–+0.037); cuti_beuti_minus_oisst -0.010 (CI -0.043–+0.022).

## 3. 조사예산 곡선

- 현재 상태 모델, 상위 5% 조사: 사건 회수율 8.8% (95% year-block CI 7.6%–10.5%).
- 현재 상태 모델, 상위 10% 조사: 사건 회수율 17.0% (95% year-block CI 14.9%–19.8%).
- 현재 상태 모델, 상위 20% 조사: 사건 회수율 29.7% (95% year-block CI 27.3%–32.8%).
- 현재 상태 모델, 상위 30% 조사: 사건 회수율 40.9% (95% year-block CI 38.6%–43.9%).

## 4. 라벨·최소 캐노피 민감도

- 20/30/40% 급감 × 최소 캐노피 0.02/0.05/0.10의 9개 조합에서 현재 상태 macro within-year AP lift 범위는 0.138–0.163였다.
- OISST 증분 추정치가 양수였던 조합은 0/9개였다. 개별 추정치와 신뢰구간은 sensitivity_increment_summary.csv에 기록했다.

## 해석 가드레일

- pooled와 within-year, random과 expanding 결과를 섞어 단일 성능처럼 보고하지 않는다.
- 모델 계열 비교는 고정된 합리적 사양의 강건성 분석이며 최고 알고리즘 선발전이 아니다.
- 30%는 생태 붕괴 임계값이 아니라 운영 라벨이다.
- OISST/CUTI/BEUTI 결과는 선택한 공개 proxy의 증분 예측정보에 관한 것이며 환경의 생태적 중요성을 부정하지 않는다.
