# Kelpwatch Cell-Level Export Automation Investigation

This note summarizes the Stage 1 investigation of Kelpwatch cell-level CSV export automation.

## Test Scope

Only the first three fishnet cells were tested:

```text
cell_001.geojson
cell_002.geojson
cell_003.geojson
```

The full 285-cell export was not run.

## Observed Web App Request Pattern

Static inspection of the Kelpwatch production JavaScript bundle showed that the web app uses a two-step request flow for uploaded geometries.

### 1. Geometry Upload

Request URL:

```text
https://shp2json.codefornature.org/api/upload?multi=true
```

Request method:

```text
POST
```

Payload:

```text
multipart/form-data
field name: geoupload
file: single-feature GeoJSON
```

The GeoJSON is sent directly as an uploaded file in the multipart request.

Observed response:

```json
{"id": "UPLOAD_ID"}
```

No authentication, CSRF token, or session cookie was required in the 3-cell test.

### 2. CSV Aggregate Download

Request URL:

```text
https://kelp-production-agg.kelpwatch.org/aggregate/id
```

Request method:

```text
GET
```

Query parameters:

```text
id=<UPLOAD_ID>
start=1984
end=2026
source=landsat
```

The Kelpwatch UI does not send a separate `timeFilter=max` parameter for CSV downloads. Instead, the CSV includes quarterly rows plus `quarter = max` rows. The modeling workflow should filter to `quarter = max` after download.

Observed response:

```text
Content-Type: text/csv; charset=utf-8
```

Required CSV columns observed:

```text
year
quarter
kelp_area_m2
count_cells_kelp
count_cells_no_clouds
count_cells_historic_footprint
```

## 3-Cell Test Result

Direct API requests succeeded for:

```text
cell_001
cell_002
cell_003
```

Each returned HTTP 200 for upload and HTTP 200 for CSV download. The CSV responses contained the required columns and `quarter = max` rows.

## Automation Decision

Direct API automation is practical for this workflow. The repository therefore includes:

```text
scripts/download_kelpwatch_cell_exports.py
scripts/validate_kelpwatch_exports.py
```

The download script defaults to cells 001-003 only. The full 285-cell run should not be executed until the 3-cell test is reviewed and confirmed.
