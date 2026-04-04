# Performance Data Uploads

Place evaluation CSV files under `performance-data/` so the KPI dashboard generator can discover them automatically.

Supported naming conventions:

- `performance-data/<dataset>/<commit>.csv`
- `performance-data/<dataset>__<commit>.csv`
- `performance-data/<dataset>--<commit>.csv`

Where:

- `<dataset>` is any dataset name such as `eth3d_nointr`
- `<commit>` is the evaluated Git commit hash, with 7 to 40 hexadecimal characters

CSV expectations:

- The header must include `scope` and `scene`
- AUC metrics are discovered dynamically from columns whose names start with `AUC`
- Per-scene rows should use `scope=scene`
- A summary row can use `scope=summary_avg`

Example:

- `performance-data/eth3d_nointr/1a2b3c4.csv`
