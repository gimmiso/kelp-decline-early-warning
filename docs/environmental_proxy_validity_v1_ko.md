# 환경변수 proxy 타당성 감사

## 결론

OISST와 CUTI/BEUTI를 켈프 연구의 환경 proxy로 사용하는 것 자체는 선행연구 관행에 부합한다. 다만 이 자료들은 10 km 셀의 현장 환경이나 직접 원인 측정값이 아니다. 본 연구에서는 **광역 환경 proxy가 캐노피 이력 이후 추가 예측가치를 제공하는지**만 평가하며, 인과효과·현장 수온·실측 영양염으로 표현하지 않는다.

## 변수별 판정

| 변수 | 대표하는 construct | 선행연구 관행 | 본 연구의 허용 표현 | 금지 표현 | 적용범위와 핵심 한계 |
|---|---|---|---|---|---|
| NOAA OISST v2.1 | 지역 표층 열노출 | 0.25° OISST를 100 km 해안구간의 SST anomaly, warmest month, heatwave days로 집계한 giant kelp 연구가 있다 | regional surface thermal-exposure proxy | 10 km 셀의 현장 수온, 저층 수온, 직접 생리 스트레스 | 주간 0.25°; 최근접 유효 해양점 최대 이동 27.8 km; 본 연구의 10 km 공간단위보다 거침 |
| CUTI | 연안대 수직 체적수송 | CCS의 용승/침강 수송지수로 널리 사용 | coastal upwelling-transport proxy | 셀별 실제 용승량 | U.S. West Coast 31–47°N, 1° 위도 bin, 해안에서 약 75 km까지 통합 |
| BEUTI | 표층 혼합층으로의 수직 질산염 플럭스 추정 | bull kelp 연구에서 겨울·봄 nutrient-availability proxy로 사용 | nitrate-flux / nutrient-supply proxy | 실측 질산염 농도, 켈프가 실제 흡수한 영양염 | CUTI와 모델 기반 subsurface nitrate의 곱; 온도–위도–질산염 관계의 정상성 가정 포함 |

## 선행연구에서 실제로 어떻게 집계했는가

### Giant kelp와 OISST

Cavanaugh 등(2019)은 0.25° 일별 OISST를 100 km 해안구간에 공간 평균하고, 월 anomaly, heatwave days, warmest-month SST를 계산했다. 가장 따뜻한 달의 절대 SST가 heatwave 저항성과 가장 강하게 연관됐고, 회복력은 SST 지표로 설명되지 않았다. 따라서 본 연구도 평균 anomaly만 쓰지 않고 절대 고온·고온빈도·계절 anomaly를 함께 사용하되, 회복이나 기작을 설명한다고 주장하지 않는다.

- Cavanaugh et al. (2019), *Spatial Variability in the Resistance and Resilience of Giant Kelp in Southern and Baja California to a Multiyear Heatwave*. DOI: https://doi.org/10.3389/fmars.2019.00413

### Bull kelp와 계절 SST·BEUTI

García-Reyes 등(2022)은 여름 bull kelp canopy에 대해 JFM 겨울 SST와 AMJ 봄 BEUTI를 평가했다. 같은 해 겨울·봄 신호가 여름 canopy와 연관됐지만, 붕괴 이후에는 환경모형의 예측력이 크게 약해졌고 성게 등 미측정 생물과정의 중요성을 지적했다. 이는 환경 proxy가 생태적으로 정당하지만, canopy history를 넘어 항상 안정적인 예측값을 주지는 않는다는 직접적 선례다.

- García-Reyes et al. (2022), *Winter oceanographic conditions predict summer bull kelp canopy cover in northern California*. DOI: https://doi.org/10.1371/journal.pone.0267737

### CUTI/BEUTI의 원래 의미와 support

Jacox 등(2018)에 따르면 CUTI는 수직 체적수송, BEUTI는 수직 질산염 플럭스의 추정치다. 두 지수는 31–47°N U.S. West Coast에 대해 1° 위도구간으로 계산되며, 국지적인 cross-shore 구조를 표현하지 않는다. BEUTI의 질산염 성분도 온도–위도–질산염 통계관계에 의존한다. 따라서 Baja에 값을 복제하거나 47°N 북쪽 Washington에 47° 값을 대입하는 것은 허용하지 않는다.

- Jacox et al. (2018), *Coastal Upwelling Revisited: Ekman, Bakun, and Improved Upwelling Indices for the U.S. West Coast*. DOI: https://doi.org/10.1029/2018JC014187

## 이번 실험에 적용한 규칙

1. OISST는 165셀 전역에서 사용하되 `regional thermal-exposure proxy`로만 해석한다.
2. 완전한 1984–2024 주간 OISST를 사용한다. 갱신이 2024-11-27에서 멈추고 1994–1996이 누락된 기존 AOML daily mirror는 폐기했다.
3. 고정된 1984–2004 baseline으로 anomaly와 90백분위 고온주를 계산한다.
4. 열 특징은 annual anomaly, warmest weekly SST, hot weeks, JFM winter anomaly, JAS summer anomaly, lag-1 anomaly로 제한한다.
5. CUTI/BEUTI는 U.S. 31–47°N에 해당하는 107셀만 사용한다. Baja와 47°N 북쪽은 제외하며 clamp하지 않는다.
6. upwelling 특징은 JFM winter, AMJ spring, March–September upwelling-season 집계와 1988–2004 baseline anomaly를 사용한다.
7. OISST의 증분가치는 165셀 동일 행에서 `trajectory+OISST − trajectory`로 계산한다.
8. CUTI/BEUTI의 순수 증분가치는 107셀 동일 행에서 `trajectory+OISST+CUTI/BEUTI − trajectory+OISST`로 계산한다.

## 해석 문장

허용:

> 공개 NOAA 기반의 광역 열·용승·질산염 플럭스 proxy는 캐노피 이력 기준선 이후 안정적인 추가 순위화 성능을 제공하지 않았다.

금지:

> 수온과 영양염은 켈프 급감에 중요하지 않았다.

환경 블록이 무효라는 결과는 환경과정의 생태적 무효가 아니라, 선택한 공개 proxy·해상도·예측시점에서의 **추가 예측정보 부재**를 뜻한다.
