# MUR 1 km 접근·전송 감사

## 고정 자료

- 자료: NASA JPL MUR v4.1 `analysed_sst`.
- 운영 endpoint: NOAA CoastWatch ERDDAP `jplMURSST41`.
- site 매핑: 183개 site를 유효한 최근접 해양 pixel에 매핑했다.
- 고유 source pixel: 146개.
- 최대 매칭거리: 0.925 km; 95백분위: 0.716 km.
- 사용기간: 2002-06-01–2021-12-31.

## 실패·대안 기록

1. **단일 pixel 20년 CSV/NetCDF**: 120–240초 timeout으로 반복돼 사용하지 않았다.
2. **단일 pixel 5년 CSV**: 일부 첫 조각은 성공했으나 이후 timeout이 반복돼 사용하지 않았다.
3. **단일 pixel 연도별 CSV**: 안정적이지만 2,920개 요청이 필요해 최종 방식으로 채택하지 않았다.
4. **0.2° tile-year**: 일부 타일의 불필요한 사각영역이 커져 응답이 느려 사용하지 않았다.
5. **AWS Open Data `s3://mur-sst/zarr-v1/`**: 로그인 없이 접근 가능하지만 공개 registry의 시간범위가 2020-01-20까지이고 chunk가 `(5, 1799, 3600)`이어서 2002–2021 연안 point 시계열 추출에 부적합했다.
6. **Google Earth Engine `JPL/MURSST/v4.1`**: 로컬 사용자 인증은 있었지만 이 작업에 사용할 등록 Cloud project가 없어 실행하지 않았다. 새 권한이나 project를 임의 생성하지 않았다.
7. **NOAA `upwell` host**: 별도 IP를 반환했지만 자료 요청이 `coastwatch` host로 redirect되어 독립 복제본으로 세지 않았다.

## 최종 전송 규격

- 분석 source pixel은 site별 최근접 0.01° MUR pixel로 끝까지 고정한다. 전송만 2002–2007은 기존 0.1° 묶음(61개), 2008–2021은 0.2° 묶음(45개)으로 처리하며, 각 사각형에서 필요한 source pixel만 즉시 추출한다.
- 2002–2021을 연도별 20개 기간으로 나눈다. 2–3년 묶음은 일부 연안 타일에서 장시간 응답과 재시도가 잦았고, 연도별 요청이 가장 안정적이었다. 시험 중 완성된 장기·연도 캐시는 날짜와 키를 검증해 재사용한다.
- 각 tile-period의 bounding rectangle을 한 번 요청하고 필요한 source pixel만 즉시 추출한다.
- 추출된 point-day만 `data/external/noaa/cache/mur1km_giraldo_v1/tile_year_points/`에 저장한다.
- NOAA CoastWatch, Brown RI Data Discovery Center, USF Marine Science ERDDAP의 동일 `jplMURSST41` 복제본을 사용한다.
- 실행 전 세 endpoint에서 동일 pixel·날짜의 값을 비교하며 허용 최대 절대차는 `1e-9 °C`이다. 검사를 통과하지 못하면 실행을 중단한다.
- 동시에 서로 다른 tile이 처리되도록 작업 큐를 period→tile 순서로 구성하고, endpoint별 worker pool을 분리한다.
- 총 병렬도는 18(각 endpoint 6)로 제한한다. 32(실질적으로 NOAA에 중복 연결)는 처리량을 높이지 못해 채택하지 않았고, HTTP 429와 일시 오류는 지수형 대기 후 다음 복제본으로 전환해 재시도한다.
- 완성된 tile-period 캐시는 재실행 시 검증 후 재사용한다.

## 해석 가드레일

- 전송방식 변경은 값·공간 source pixel·기간·특징정의를 바꾸지 않는다.
- MUR는 표층수온이며 Giraldo의 해저수온과 동일하지 않다.
- MUR 증분가치가 없더라도 수온의 생태적 무관성을 의미하지 않는다.
