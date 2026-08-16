# 관측품질·라벨 강건성 독립 검증

## Overall Assessment: Share with caveats

prediction 수준 재계산, 시계열 누수, 동일행 비교, 고품질 부분표본, 고정 footprint 재현 및 파일 hash를 다시 확인했다. 총 20개 검사 중 20개가 통과했고, 공유를 막는 오류는 없다.

## 재계산 결과

- 연도별 AP lift 최대 저장 오차: 1.11e-16
- macro AP lift 최대 저장 오차: 8.33e-17
- Top-20% recall 최대 저장 오차: 5.55e-17
- 저장 bootstrap draw에서 2.5%·97.5% 구간 재계산 최대 오차: 9.02e-17
- common-row 모드는 8개 라벨 정의에서 같은 cell-year를 사용했다.
- 고관측품질 결과행은 주 native 결과행의 진부분집합이다.
- 공식 NetCDF에서 다시 만든 pre-2005 footprint pixel 수와 기존 inventory가 165셀 모두 일치했다.

## 해석상 필수 제한

- 이 분석은 기존 결과를 확인한 뒤 수행한 강건성 분석이며 독립 확증이 아니다.
- `area_se`는 픽셀 공분산과 annual-maximum 선택오차가 없어 inverse-variance 가중치로 쓰지 않는다.
- 센서시대별 성능 차이는 관측기회·사건구성·생태상태가 함께 달라진 기술적 진단이며 센서의 인과효과가 아니다.
- Kelpwatch 웹/API와 지역별 층화표본 대조는 아직 별도 Gate 2로 남아 있다.
