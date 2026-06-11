# Kelpwatch Decline Label Report

## Summary

Input rows: 2100
Number of cells: 50
Year range: 1984-2025
Baseline period: 1984-2013
Rows with valid next-year labels: 2050
Main decline-event count: 696
Main decline-event rate: 0.3395

## Robustness Label Counts

Full-history p25 decline-event count: 513
Full-history p25 decline-event rate: 0.2502
50 percent next-year decline count: 517
50 percent next-year decline rate: 0.2522

## Decline-Event Count by Region

- Central California: 417 / 1394 rows (0.2991)
- Northern California: 279 / 656 rows (0.4253)

## Decline-Event Count by Year

- 1984: 7 / 50 rows (0.1400)
- 1985: 10 / 50 rows (0.2000)
- 1986: 12 / 50 rows (0.2400)
- 1987: 7 / 50 rows (0.1400)
- 1988: 2 / 50 rows (0.0400)
- 1989: 1 / 50 rows (0.0200)
- 1990: 15 / 50 rows (0.3000)
- 1991: 9 / 50 rows (0.1800)
- 1992: 16 / 50 rows (0.3200)
- 1993: 7 / 50 rows (0.1400)
- 1994: 35 / 50 rows (0.7000)
- 1995: 24 / 50 rows (0.4800)
- 1996: 7 / 50 rows (0.1400)
- 1997: 43 / 50 rows (0.8600)
- 1998: 5 / 50 rows (0.1000)
- 1999: 13 / 50 rows (0.2600)
- 2000: 2 / 50 rows (0.0400)
- 2001: 3 / 50 rows (0.0600)
- 2002: 25 / 50 rows (0.5000)
- 2003: 9 / 50 rows (0.1800)
- 2004: 21 / 50 rows (0.4200)
- 2005: 26 / 50 rows (0.5200)
- 2006: 2 / 50 rows (0.0400)
- 2007: 4 / 50 rows (0.0800)
- 2008: 0 / 50 rows (0.0000)
- 2009: 14 / 50 rows (0.2800)
- 2010: 26 / 50 rows (0.5200)
- 2011: 12 / 50 rows (0.2400)
- 2012: 0 / 50 rows (0.0000)
- 2013: 31 / 50 rows (0.6200)
- 2014: 21 / 50 rows (0.4200)
- 2015: 21 / 50 rows (0.4200)
- 2016: 29 / 50 rows (0.5800)
- 2017: 19 / 50 rows (0.3800)
- 2018: 35 / 50 rows (0.7000)
- 2019: 26 / 50 rows (0.5200)
- 2020: 23 / 50 rows (0.4600)
- 2021: 24 / 50 rows (0.4800)
- 2022: 42 / 50 rows (0.8400)
- 2023: 37 / 50 rows (0.7400)
- 2024: 31 / 50 rows (0.6200)

## Label Definitions

The main early-warning target is `decline_event_next`, which equals 1 when the following year's `relative_canopy` falls below the cell-specific 25th percentile of `relative_canopy` during the 1984-2013 baseline period.

Robustness labels include `decline_event_next_p25_full`, based on each cell's full-history 25th percentile, and `decline_50pct_next`, based on a next-year canopy decline of at least 50 percent from the current year.
