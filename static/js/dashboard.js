(function () {
  const MANIFEST_URL = "./static/data/performance-manifest.json";
  const AVERAGE_KEY = "__average__";

  const state = {
    manifest: null,
    dataset: null,
    metric: AVERAGE_KEY,
  };

  const els = {
    datasetSelect: document.getElementById("dataset-select"),
    metricSelect: document.getElementById("metric-select"),
    latestValue: document.getElementById("latest-value"),
    latestContext: document.getElementById("latest-context"),
    deltaValue: document.getElementById("delta-value"),
    deltaContext: document.getElementById("delta-context"),
    commitCount: document.getElementById("commit-count"),
    commitRange: document.getElementById("commit-range"),
    trendChart: document.getElementById("trend-chart"),
    matrixWrap: document.getElementById("matrix-wrap"),
    emptyState: document.getElementById("empty-state"),
  };

  init();

  async function init() {
    try {
      const response = await fetch(MANIFEST_URL, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Manifest request failed with ${response.status}`);
      }
      state.manifest = await response.json();
      setupControls();
      render();
    } catch (error) {
      console.error(error);
      els.emptyState.hidden = false;
    }
  }

  function setupControls() {
    const datasets = Object.keys(state.manifest.datasets || {});
    if (!datasets.length) {
      els.emptyState.hidden = false;
      return;
    }

    state.dataset = datasets[0];
    els.datasetSelect.innerHTML = datasets
      .map((dataset) => `<option value="${escapeHtml(dataset)}">${escapeHtml(dataset)}</option>`)
      .join("");

    els.datasetSelect.addEventListener("change", (event) => {
      state.dataset = event.target.value;
      state.metric = AVERAGE_KEY;
      syncMetricOptions();
      render();
    });

    els.metricSelect.addEventListener("change", (event) => {
      state.metric = event.target.value;
      render();
    });

    syncMetricOptions();
  }

  function syncMetricOptions() {
    const dataset = getActiveDataset();
    const options = [{ value: AVERAGE_KEY, label: "Average over AUCs" }]
      .concat((dataset.metrics || []).map((metric) => ({ value: metric, label: metric })));

    els.metricSelect.innerHTML = options
      .map((option) => `<option value="${escapeHtml(option.value)}">${escapeHtml(option.label)}</option>`)
      .join("");
    els.metricSelect.value = state.metric;
  }

  function render() {
    const dataset = getActiveDataset();
    if (!dataset) {
      els.emptyState.hidden = false;
      return;
    }

    els.emptyState.hidden = true;
    renderStats(dataset);
    renderTrendChart(dataset);
    renderMatrix(dataset);
  }

  function renderStats(dataset) {
    const commits = dataset.commits || [];
    const latest = commits[commits.length - 1];
    const previous = commits[commits.length - 2];
    const latestValue = latest ? getMetricValue(latest.summary, state.metric) : null;
    const previousValue = previous ? getMetricValue(previous.summary, state.metric) : null;
    const delta = latestValue != null && previousValue != null ? latestValue - previousValue : null;

    els.latestValue.textContent = formatValue(latestValue);
    els.latestContext.textContent = latest
      ? `${formatDate(latest.committed_at)} • ${latest.short_commit}`
      : "Waiting for data";

    els.deltaValue.textContent = delta == null ? "--" : signedValue(delta);
    els.deltaValue.style.color = delta == null ? "" : delta >= 0 ? "rgb(var(--gain))" : "rgb(var(--loss))";
    els.deltaContext.textContent = previous ? `Compared with ${previous.short_commit}` : "No previous commit";

    els.commitCount.textContent = String(commits.length);
    els.commitRange.textContent = commits.length
      ? `${formatDate(commits[0].committed_at)} to ${formatDate(latest.committed_at)}`
      : "No history yet";
  }

  function renderTrendChart(dataset) {
    const commits = dataset.commits || [];
    if (!commits.length) {
      els.trendChart.innerHTML = "<p class=\"panel-note\">No commits available yet.</p>";
      return;
    }

    const width = Math.max(760, commits.length * 120);
    const height = 340;
    const padding = { top: 20, right: 24, bottom: 78, left: 58 };
    const values = commits.map((commit) => getMetricValue(commit.summary, state.metric)).filter(isFiniteNumber);
    const minValue = values.length ? Math.min(...values) : 0;
    const maxValue = values.length ? Math.max(...values) : 100;
    const span = Math.max(maxValue - minValue, 5);
    const domainMin = Math.max(0, minValue - span * 0.12);
    const domainMax = maxValue + span * 0.12;

    const x = (index) => {
      if (commits.length === 1) {
        return width / 2;
      }
      const usable = width - padding.left - padding.right;
      return padding.left + (usable * index) / (commits.length - 1);
    };

    const y = (value) => {
      const usable = height - padding.top - padding.bottom;
      return padding.top + ((domainMax - value) / (domainMax - domainMin || 1)) * usable;
    };

    const gridTicks = buildTicks(domainMin, domainMax, 5);
    const plottedPoints = commits
      .map((commit, index) => ({
        commit,
        index,
        value: getMetricValue(commit.summary, state.metric),
      }))
      .filter((item) => isFiniteNumber(item.value));

    if (!plottedPoints.length) {
      els.trendChart.innerHTML = "<p class=\"panel-note\">No summary values are available for the selected metric.</p>";
      return;
    }

    const path = plottedPoints
      .map((item, pointIndex) => {
        const command = pointIndex === 0 ? "M" : "L";
        return `${command} ${x(item.index).toFixed(2)} ${y(item.value).toFixed(2)}`;
      })
      .join(" ");

    const gridLines = gridTicks.map((tick) => {
      const tickY = y(tick);
      return `
        <line class="grid-line" x1="${padding.left}" y1="${tickY}" x2="${width - padding.right}" y2="${tickY}"></line>
        <text class="axis-label" x="${padding.left - 10}" y="${tickY + 4}" text-anchor="end">${formatValue(tick)}</text>
      `;
    }).join("");

    const xLabels = commits.map((commit, index) => `
      <text class="axis-label" x="${x(index)}" y="${height - 34}" text-anchor="middle">${escapeHtml(commit.short_commit)}</text>
      <text class="axis-label" x="${x(index)}" y="${height - 18}" text-anchor="middle">${escapeHtml(formatDate(commit.committed_at, true))}</text>
    `).join("");

    const points = plottedPoints.map((item) => {
      const cx = x(item.index);
      const cy = y(item.value);
      const title = `${item.commit.short_commit} • ${formatDate(item.commit.committed_at)} • ${formatValue(item.value)}`;
      return `
        <g>
          <title>${escapeHtml(title)}</title>
          <circle class="trend-point" cx="${cx}" cy="${cy}" r="6"></circle>
        </g>
      `;
    }).join("");

    els.trendChart.innerHTML = `
      <svg class="trend-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Commit trend chart">
        ${gridLines}
        <line class="grid-line" x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}"></line>
        <path class="trend-line" d="${path}"></path>
        ${points}
        ${xLabels}
      </svg>
    `;
  }

  function renderMatrix(dataset) {
    const commits = dataset.commits || [];
    const scenes = dataset.scenes || [];
    if (!commits.length || !scenes.length) {
      els.matrixWrap.innerHTML = "<p class=\"panel-note\">No per-scene data available yet.</p>";
      return;
    }

    const header = commits.map((commit) => `
      <th class="commit-label">
        <strong>${escapeHtml(commit.short_commit)}</strong>
        <span>${escapeHtml(formatDate(commit.committed_at, true))}</span>
      </th>
    `).join("");

    const rows = scenes.map((scene) => {
      const cells = commits.map((commit, index) => {
        const current = getMetricValue(commit.scenes[scene], state.metric);
        const previous = index > 0 ? getMetricValue(commits[index - 1].scenes[scene], state.metric) : null;
        const delta = current != null && previous != null ? current - previous : null;
        const background = delta == null ? "rgba(var(--neutral), 0.08)" : colorForDelta(delta);
        const title = delta == null
          ? `${scene} • ${commit.short_commit} • ${formatValue(current)}`
          : `${scene} • ${commit.short_commit} • ${formatValue(current)} (${signedValue(delta)} vs previous)`;
        return `
          <td class="matrix-cell" style="background:${background}" title="${escapeHtml(title)}">
            ${formatValue(current)}
          </td>
        `;
      }).join("");

      return `
        <tr>
          <th class="scene-name">${escapeHtml(scene)}</th>
          ${cells}
        </tr>
      `;
    }).join("");

    els.matrixWrap.innerHTML = `
      <table class="matrix-table">
        <thead>
          <tr>
            <th>Scene</th>
            ${header}
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  function getActiveDataset() {
    return state.manifest && state.manifest.datasets ? state.manifest.datasets[state.dataset] : null;
  }

  function getMetricValue(record, metric) {
    if (!record) {
      return null;
    }
    const value = record[metric];
    return isFiniteNumber(value) ? value : null;
  }

  function buildTicks(min, max, count) {
    const ticks = [];
    const step = (max - min) / count;
    for (let index = 0; index <= count; index += 1) {
      ticks.push(min + step * index);
    }
    return ticks;
  }

  function colorForDelta(delta) {
    const capped = Math.min(Math.abs(delta) / 10, 1);
    const alpha = 0.12 + capped * 0.3;
    const rgb = delta >= 0 ? "var(--gain)" : "var(--loss)";
    return `rgba(${rgb}, ${alpha})`;
  }

  function formatValue(value) {
    return value == null ? "--" : Number(value).toFixed(2);
  }

  function signedValue(value) {
    return value == null ? "--" : `${value >= 0 ? "+" : ""}${Number(value).toFixed(2)}`;
  }

  function formatDate(value, compact) {
    if (!value) {
      return compact ? "--" : "Unknown date";
    }
    const date = new Date(value);
    return new Intl.DateTimeFormat("en-US", compact
      ? { month: "short", day: "numeric", year: "2-digit" }
      : { year: "numeric", month: "short", day: "numeric" }).format(date);
  }

  function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll("\"", "&quot;")
      .replaceAll("'", "&#39;");
  }
})();
