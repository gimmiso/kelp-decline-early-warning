# California local-support panel 독립 검증

- 판정: **Share with caveats**
- 통과: 9/9
- raw NetCDF spot checks: 18/18

300 m와 1 km에서 각각 세 site, 2005·2014·2021년을 골라 고정 pre-2005 positive footprint, 분기 유효성, annual maximum과 선택분기를 공식 NetCDF에서 다시 계산했다.

| check                                     | passed   | severity_if_failed   | evidence                   |
|:------------------------------------------|:---------|:---------------------|:---------------------------|
| panel_keys_unique                         | True     | blocker              | 0                          |
| metadata_keys_unique                      | True     | blocker              | 0                          |
| field_keys_unique                         | True     | blocker              | 0                          |
| complete_site_year_rectangle              | True     | blocker              | expected=14700; rows=14700 |
| panel_contains_exact_stable_support_sites | True     | blocker              | metadata=350; panel=350    |
| event_rule_recomputed                     | True     | blocker              | 0                          |
| eligibility_rule_recomputed               | True     | blocker              | 0                          |
| reference_p95_recomputed                  | True     | blocker              | max_error=1.16e-10         |
| raw_netcdf_spot_checks_match              | True     | blocker              | passed=18/18               |
