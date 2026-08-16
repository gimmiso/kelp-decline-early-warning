# 저널 보완 실험 검증 보고서

## Overall Assessment: Share with caveats

총 28개 검증 중 28개가 통과했다. 공유를 막는 계산·누수·행 불일치 오류는 없다.

## Methodology Review

- 운영 성능은 2005–2024 expanding-window와 연도 내 AP lift로 평가했다.
- random cell-year CV는 동일한 2005–2024 평가행을 사용하지만 미래 연도와 동일 셀의 다른 연도를 학습에 허용한다. 이는 낙관 편향을 보여주기 위한 비교이며 배포성능 추정치가 아니다.
- 모델 계열은 결과를 본 뒤 최고 사양을 선택하지 않고 Logistic, Random Forest, XGBoost의 고정 사양을 사용했다.
- 라벨 민감도는 20/30/40% 감소와 현재 상대 캐노피 0.02/0.05/0.10의 9개 조합을 모두 expanding-window로 다시 적합했다.

## Calculation Spot-Checks

- 연도별 AP lift 최대 재계산 오차: 9.71e-17.
- 모델 macro AP lift 최대 오차: 8.33e-17.
- 정보 블록 증분값 최대 오차: 9.28e-17.
- random-CV repeat 0의 pooled/macro 최대 오차: 8.33e-17.
- 예산별 선발 수·적중 사건 수 최대 정수 오차: 0.
- 민감도 평가행 수 최대 오차: 0; 사건 수 최대 오차: 0.
- 입력 및 핵심 산출물 hash가 manifest와 일치한다.

## Issues Found

1. **Medium — 확증 수준:** 이전 결과를 확인한 뒤 수행한 강건성 분석이므로 독립 확증이나 사전등록으로 표현할 수 없다.
2. **Medium — random-CV 해석:** random-CV와 expanding-window의 차이는 시간 누수, 동일 셀 반복, 학습량 차이가 결합된 결과다. 특정 한 원인의 인과효과로 분해하지 않는다.
3. **Medium — 모델 비교 범위:** 세 모델은 고정 사양의 결론 강건성 점검이며 최적 알고리즘 선발이나 완전한 하이퍼파라미터 탐색이 아니다.
4. **Medium — 라벨 타당성:** 민감도 통과는 30%가 생태학적 붕괴 임계값임을 증명하지 않는다. 모두 annual-maximum canopy의 운영 라벨이다.
5. **Medium — 환경 해석:** NOAA 결과는 선택한 proxy의 증분 예측정보에 관한 것이며 환경의 생태적 중요성이나 인과효과를 검정하지 않는다.

## Required Caveats for Paper

- 현재 상태 신호는 세 모델에서 반복됐다고 쓸 수 있다.
- 궤적 증분은 세 고정 모델 모두에서 양수로 지지됐으므로 ‘작지만 모델 계열에 강건한 추가가치’로 표현할 수 있다.
- OISST와 CUTI/BEUTI는 세 모델 모두에서 양의 증분가치가 확정되지 않았다고 쓸 수 있다.
- random split이 항상 모든 지표를 부풀렸다고 쓰면 안 된다. 현재 상태의 macro within-year 지표는 random과 expanding이 거의 같았고, 복잡한 정보 블록과 pooled 지표에서 낙관 차이가 커졌다.
