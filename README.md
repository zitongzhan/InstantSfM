# InstantSfM

This is the repository that contains source code for the [InstantSfM website](https://cre185.github.io/InstantSfM/).

## KPI dashboard

The repository now also includes a static KPI dashboard at `dashboard.html` for commit-by-commit evaluation tracking.

Performance CSVs placed under `performance-data/` are converted into `static/data/performance-manifest.json` by `scripts/generate_performance_manifest.py`. The GitHub Actions workflow at `.github/workflows/deploy-dashboard.yml` regenerates that manifest and deploys the site whenever dashboard assets or performance uploads change.

# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
