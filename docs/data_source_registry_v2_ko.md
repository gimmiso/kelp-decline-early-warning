# 데이터 출처 및 적합성 Registry v2

이 문서는 `KELP-EWS-E2E-v2.0`에서 고려하는 데이터의 역할과 적용범위를 고정한다. 날짜와 범위는 2026-08-13 확인 기준이며 실제 실행 시 snapshot hash를 추가한다.

| Source | 공식 범위 | 공간·시간 support | 주 역할 | 현재 결정 | 핵심 위험 |
|---|---|---|---|---|---|
| Kelpwatch/EDI `knb-lter-sbc.74.34` | Baja California–U.S./Canada border | 약 30 m pixels, quarterly, 1984–present | canopy outcome/history | 주 자료 | all-history footprint, cloud/sensor effort, species 미구분 |
| NOAA CRW CoralTemp v3.1 | global | 0.05°, daily, 1985–present | full-domain thermal exposure | **주 NOAA 열 자료** | nearshore support, high-latitude anomaly 신뢰도, coral-derived ancillary thresholds |
| NOAA OISST v2.1 | global | 0.25°, daily, 1981–present | legacy/coarse thermal sensitivity | 보조 | 10 km kelp cell보다 거친 공간 support |
| NOAA CUTI | U.S. West Coast, 31–47°N | 1° latitude bins, daily, 1988–present | upwelling transport proxy | 지원범위 subset | cell-specific 현장값 아님, 범위 밖 외삽 금지 |
| NOAA BEUTI | U.S. West Coast, 31–47°N | 1° latitude bins, daily, 1988–present | nitrate-flux proxy | 지원범위 subset | cell-specific nutrient 측정 아님, 범위 밖 외삽 금지 |
| Kelpwatch Planet | California | 3 m, September, 2017–2024 | recent outcome measurement audit | 탐색 | 짧은 기간, Landsat와 시간집계 다름 |
| ECMWF ERA5 ocean waves | global | reanalysis, 1940–present; wave grid is coarser than kelp cells | full-domain wave disturbance proxy | **coverage gate 후 잠금 보조** | 근해/섬 support, local breaking wave가 아님 |
| CDIP MOP/hindcast | California | nearshore modeled wave series, site/model dependent | ERA5 coastal-support audit | 품질 대조 | Baja–Washington 전역의 동일기간 자료가 아님 |
| bathymetry/habitat | source-specific | static | spatial context sensitivity | 탐색 | 동적 조기신호 아님 |

## 확인한 공식 근거

- Kelpwatch methodology: https://kelpwatch.org/methodology
- Kelpwatch GIS download page: https://kelpwatch.org/
- EDI metadata: https://portal.edirepository.org/nis/metadataviewer?packageid=knb-lter-sbc.74.34
- NOAA OISST v2.1: https://www.psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
- NOAA CRW 5 km methodology: https://coralreefwatch.noaa.gov/product/5km/methodology.php
- NOAA CUTI metadata: https://upwell.pfeg.noaa.gov/erddap/info/erdCUTIdaily/index.html
- NOAA BEUTI metadata: https://upwell.pfeg.noaa.gov/erddap/info/erdBEUTIdaily/index.html
- ECMWF ERA5 reanalysis catalogue: https://www.ecmwf.int/en/forecasts/datasets/browse-reanalysis-datasets
- NOAA NCEI WAVEWATCH III hindcast metadata: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc%3ANCEP-WAVEWATCH
- CDIP California wave models: https://www.cdip.ucsd.edu/m/documents/models.html

## 현재 확인된 Kelpwatch snapshot

- File: `LandsatKelpBiomass_2026_Q2_withmetadata.nc`
- Package: `knb-lter-sbc.74.34`
- DOI: `10.6073/pasta/5b441f39c0876ef901068709df8f3e50`
- Created/modified: 2026-08-04
- SHA-256: `b27caf8e4d08c0d2816d044e254a6339c854f2933e14a3fa2a139b37832f884f`
- Time: 1984-01-01–2026-06-30, 170 quarters
- Coordinates: 593,427 stations; valid 593,426; invalid 1
- Bounds: 27.0088–48.3953°N, 124.7666–114.0420°W
- License: CC BY 4.0

## 해결해야 할 source-level 쟁점

1. EDI NetCDF의 title/summary는 central/southern California라고 쓰지만 좌표와 Kelpwatch 웹문서는 더 넓은 West Coast 범위를 가리킨다. 데이터 담당자 문서 또는 EDI XML로 확인한다.
2. `count_cells_historic_footprint`는 전 관측기간 중 한 번이라도 kelp가 있었던 pixel 수이므로 cohort와 predictor normalization에 직접 쓰지 않는다.
3. CRW의 SST 자체는 사용 가능하지만 coral-specific HotSpot/DHW 임계값을 kelp 생태 임계값으로 재해석하지 않는다.
4. CUTI/BEUTI는 31–47°N U.S. West Coast proxy다. 기존 코드의 31/47 clamp는 전 서해안 확장에 부적합하므로 사용하지 않는다.
5. Planet은 더 정밀하지만 기간이 짧아 주 백테스트를 대체할 수 없다.
6. 파랑은 kelp 선행연구에서 핵심 교란요인이지만, NOAA WAVEWATCH III 공개 hindcast snapshot은 2005–2019이고 근해 미세격자 적용에 제한이 명시돼 있어 1984–2024 전역 주 자료로는 부적합하다. ERA5를 장기 전역 후보로 감사하되, 결과를 보기 전에 coverage gate로 채택/제외한다.

## 실행 시 추가할 필드

각 다운로드마다 아래를 immutable manifest에 기록한다.

```text
source_id
provider
dataset_version
retrieval_url
retrieved_at_utc
temporal_coverage
spatial_coverage
license
local_path
file_size_bytes
sha256
request_parameters
notes
```
