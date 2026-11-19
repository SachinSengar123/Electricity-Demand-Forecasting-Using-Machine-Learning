/**
 * GridCast AI - Frontend Dashboard Controller
 * Timezone: Asia/Kolkata (IST)
 * Real-time IST clock, dynamic 24-hour predictions, robust confidence intervals,
 * consistent single-source peak load metrics, configurable thresholds, and dataset audits.
 */

// Global State
let currentTab = 'overview';
let cachedOverviewData = null;
let cachedForecastData = null;
let cachedPeakData = null;
let cachedModelData = null;
let cachedDatasetData = null;

// Plotly Dark Theme Preset
const plotlyTheme = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: {
    family: 'Inter, sans-serif',
    color: '#94a3b8',
    size: 11
  },
  margin: { l: 55, r: 25, t: 30, b: 45 },
  xaxis: {
    gridcolor: 'rgba(255, 255, 255, 0.06)',
    zerolinecolor: 'rgba(255, 255, 255, 0.1)',
    tickfont: { color: '#94a3b8' }
  },
  yaxis: {
    gridcolor: 'rgba(255, 255, 255, 0.06)',
    zerolinecolor: 'rgba(255, 255, 255, 0.1)',
    tickfont: { color: '#94a3b8' },
    title: { font: { size: 12, color: '#06b6d4' } }
  }
};

// Application Initialization
document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  setupISTClock();
  setupNavigation();
  setupEventListeners();
  loadAllData();
});

// 1. Live Dynamic IST Clock
function setupISTClock() {
  const clockEl = document.getElementById('currentClock');
  const dateEl = document.getElementById('currentDateStr');

  const updateTime = () => {
    // Current IST time
    const now = new Date();
    const istOptions = { timeZone: 'Asia/Kolkata' };
    
    // Time format: HH:MM:SS
    const timeFormatter = new Intl.DateTimeFormat('en-IN', {
      ...istOptions,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    });
    clockEl.textContent = timeFormatter.format(now);

    // Readable Date format: 13 November 2026
    const dateFormatter = new Intl.DateTimeFormat('en-IN', {
      ...istOptions,
      day: 'numeric',
      month: 'long',
      year: 'numeric'
    });
    dateEl.textContent = dateFormatter.format(now);
  };

  updateTime();
  setInterval(updateTime, 1000);
}

// 2. Navigation Handling
function setupNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  const collapseBtn = document.getElementById('collapseBtn');
  const sidebar = document.getElementById('sidebar');

  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetTab = item.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });

  collapseBtn.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 300);
  });
}

function switchTab(tabId) {
  currentTab = tabId;

  document.querySelectorAll('.nav-item').forEach(item => {
    if (item.getAttribute('data-tab') === tabId) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  const titles = {
    'overview': 'Dashboard Overview',
    'forecast': '24-Hour Demand Forecasting',
    'peak-load': 'Peak Load & Critical Surge Analysis',
    'models': 'Machine Learning Model Comparison',
    'dataset': 'Dataset Management & Quality Audit',
    'about': 'About GridCast AI'
  };
  document.getElementById('pageTitle').textContent = titles[tabId] || 'Dashboard';

  document.querySelectorAll('.tab-content').forEach(section => {
    section.classList.remove('active');
  });
  const targetSection = document.getElementById(`tab-${tabId}`);
  if (targetSection) {
    targetSection.classList.add('active');
  }

  // Trigger chart renders for specific tabs
  if (tabId === 'overview') renderOverviewCharts();
  if (tabId === 'forecast') renderForecastTab();
  if (tabId === 'peak-load') renderPeakLoadTab();
  if (tabId === 'models') renderModelComparisonTab();
  if (tabId === 'dataset') renderDatasetTab();

  if (window.lucide) {
    lucide.createIcons();
  }
}

// 3. Event Listeners
function setupEventListeners() {
  document.getElementById('refreshBtn').addEventListener('click', () => {
    loadAllData(true);
  });

  document.getElementById('btnGenerateForecast').addEventListener('click', () => {
    fetchAndRenderForecast();
  });

  document.getElementById('forecastModelSelect').addEventListener('change', () => {
    fetchAndRenderForecast();
  });

  document.getElementById('tempScenarioSelect').addEventListener('change', () => {
    fetchAndRenderForecast();
  });

  document.getElementById('toggleUncertainty').addEventListener('change', () => {
    renderForecastTab();
  });

  document.getElementById('btnDownloadForecast').addEventListener('click', () => {
    const model = document.getElementById('forecastModelSelect').value;
    const tempMod = document.getElementById('tempScenarioSelect').value;
    window.location.href = `/api/forecast/download?model=${encodeURIComponent(model)}&temp_modifier=${tempMod}`;
  });

  document.getElementById('forecastTableFilter').addEventListener('change', (e) => {
    filterForecastTable(e.target.value);
  });

  // Peak analysis recompute
  document.getElementById('btnRecomputePeak').addEventListener('click', () => {
    fetchAndRenderPeakLoad();
  });

  // Retrain models
  document.getElementById('btnRetrainModels').addEventListener('click', async () => {
    const btn = document.getElementById('btnRetrainModels');
    btn.disabled = true;
    btn.innerHTML = `<div class="spinner" style="width:14px;height:14px;"></div> Retraining...`;
    try {
      const res = await fetch('/api/train-model', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        alert(data.message);
        loadAllData(true);
      } else {
        alert('Model training failed: ' + data.error);
      }
    } catch (e) {
      alert('Network error while retraining models.');
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="refresh-cw" style="width:16px;height:16px;"></i><span>Retrain All Models</span>`;
      if (window.lucide) lucide.createIcons();
    }
  });

  // Drag and Drop Upload
  const dropzone = document.getElementById('csvDropzone');
  const fileInput = document.getElementById('csvFileInput');

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleCsvUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleCsvUpload(e.target.files[0]);
    }
  });

  document.getElementById('btnResetDefaultDataset').addEventListener('click', () => {
    resetToDefaultDataset();
  });
}

// 4. Master Data Fetcher
async function loadAllData(showNotification = false) {
  showLoading('overviewChartLoading', true);
  try {
    const [ovRes, fcRes, pkRes, modRes, dsRes] = await Promise.all([
      fetch('/api/overview').then(r => r.json()),
      fetch('/api/forecast').then(r => r.json()),
      fetch('/api/peak-analysis').then(r => r.json()),
      fetch('/api/model-performance').then(r => r.json()),
      fetch('/api/dataset-info').then(r => r.json())
    ]);

    cachedOverviewData = ovRes;
    cachedForecastData = fcRes;
    cachedPeakData = pkRes;
    cachedModelData = modRes;
    cachedDatasetData = dsRes;

    // Update DEMO Badge
    const badge = document.getElementById('demoDataBadge');
    if (ovRes.is_demo_data) {
      badge.style.background = 'rgba(245, 158, 11, 0.12)';
      badge.style.borderColor = 'rgba(245, 158, 11, 0.35)';
      badge.style.color = '#fbbf24';
      badge.innerHTML = '<i data-lucide="shield-alert" style="width:14px;height:14px;"></i><span>DEMO DATA (SYNTHESIZED)</span>';
    } else {
      badge.style.background = 'rgba(16, 185, 129, 0.15)';
      badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      badge.style.color = '#34d399';
      badge.innerHTML = '<i data-lucide="check-circle" style="width:14px;height:14px;"></i><span>VERIFIED USER DATASET</span>';
    }

    // System Metadata Bar
    const meta = ovRes.system_metadata;
    document.getElementById('metaSelectedModel').textContent = ovRes.model_used + ' (Active)';
    document.getElementById('metaLastForecast').textContent = meta.last_forecast_time;
    document.getElementById('metaDatasetStatus').textContent = meta.dataset_status;
    document.getElementById('metaModelStatus').textContent = meta.model_status;

    // Update 4 Main KPI Cards
    const kpis = ovRes.kpis;
    document.getElementById('kpiCurrentDemand').textContent = Number(kpis.current_demand).toLocaleString();
    document.getElementById('kpiNextHourDemand').textContent = Number(kpis.next_hour_predicted).toLocaleString();
    document.getElementById('kpiAvg24hDemand').textContent = Number(kpis.next_24h_avg).toLocaleString();
    document.getElementById('kpiExpectedPeak').textContent = Number(kpis.expected_peak_demand).toLocaleString();
    document.getElementById('kpiPeakTimeLabel').textContent = `Peak expected at ${kpis.expected_peak_hour}`;

    // Render current active tab
    switchTab(currentTab);

    if (showNotification) {
      showToast('Dashboard successfully synchronized with backend.');
    }
  } catch (err) {
    console.error('Failed to load dashboard data:', err);
    showToast('Connection error: could not fetch dashboard telemetry.', 'error');
  } finally {
    showLoading('overviewChartLoading', false);
  }
}

// 5. Render Overview Charts
function renderOverviewCharts() {
  if (!cachedOverviewData) return;

  const history = cachedOverviewData.recent_history;
  const forecast = cachedOverviewData.forecast_trend;

  const histTimes = history.map(h => h.display_time);
  const histDemand = history.map(h => h.demand);

  const fcTimes = forecast.map(f => f.display_time);
  const fcDemand = forecast.map(f => f.predicted_demand);

  const traceHist = {
    x: histTimes,
    y: histDemand,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Historical Recorded Load (MW)',
    line: { color: '#38bdf8', width: 2.5 },
    marker: { size: 4, color: '#38bdf8' }
  };

  const traceFc = {
    x: fcTimes,
    y: fcDemand,
    type: 'scatter',
    mode: 'lines+markers',
    name: '24h Forecast (MW)',
    line: { color: '#06b6d4', width: 2.5, dash: 'dot' },
    marker: { size: 5, color: '#06b6d4' }
  };

  const layout = {
    ...plotlyTheme,
    title: false,
    legend: { orientation: 'h', y: 1.12, font: { color: '#cbd5e1' } },
    yaxis: { ...plotlyTheme.yaxis, title: 'Electricity Demand (MW)' },
    xaxis: { ...plotlyTheme.xaxis, title: 'Timestamp (Asia/Kolkata)' }
  };

  Plotly.newPlot('overviewTrendChart', [traceHist, traceFc], layout, { responsive: true, displayModeBar: false });
}

// 6. Fetch & Render 24-Hour Forecast
async function fetchAndRenderForecast() {
  showLoading('forecastChartLoading', true);
  const model = document.getElementById('forecastModelSelect').value;
  const tempMod = document.getElementById('tempScenarioSelect').value;

  try {
    const res = await fetch(`/api/forecast?model=${encodeURIComponent(model)}&temp_modifier=${tempMod}`);
    cachedForecastData = await res.json();
    renderForecastTab();
  } catch (err) {
    console.error('Forecast fetch failed:', err);
    showToast('Failed to fetch forecast from model service.', 'error');
  } finally {
    showLoading('forecastChartLoading', false);
  }
}

function renderForecastTab() {
  if (!cachedForecastData) return;

  const summary = cachedForecastData.summary;
  document.getElementById('fcMinDemand').textContent = Number(summary.expected_min_demand).toLocaleString();
  document.getElementById('fcMinTime').textContent = `Lowest at ${summary.expected_min_time}`;
  document.getElementById('fcAvgDemand').textContent = Number(summary.next_24h_avg).toLocaleString();
  document.getElementById('fcMaxDemand').textContent = Number(summary.expected_peak_demand).toLocaleString();
  document.getElementById('fcMaxTime').textContent = `Peak expected at ${summary.expected_peak_time}`;
  document.getElementById('tableGeneratedTimestamp').textContent = `Generated: ${cachedForecastData.generated_at}`;

  const forecast = cachedForecastData.forecast;
  const times = forecast.map(f => f.display_time);
  const values = forecast.map(f => f.predicted_demand);
  const upper = forecast.map(f => f.upper_bound);
  const lower = forecast.map(f => f.lower_bound);

  const traces = [];
  const showUncertainty = document.getElementById('toggleUncertainty').checked;

  if (showUncertainty) {
    traces.push({
      x: times,
      y: upper,
      type: 'scatter',
      mode: 'lines',
      line: { color: 'rgba(6, 182, 212, 0.05)', width: 0 },
      showlegend: false,
      hoverinfo: 'none'
    });
    traces.push({
      x: times,
      y: lower,
      type: 'scatter',
      mode: 'lines',
      fill: 'tonexty',
      fillcolor: 'rgba(6, 182, 212, 0.14)',
      line: { color: 'rgba(6, 182, 212, 0.05)', width: 0 },
      name: '95% Prediction Interval',
      showlegend: true
    });
  }

  traces.push({
    x: times,
    y: values,
    type: 'scatter',
    mode: 'lines+markers',
    name: `${cachedForecastData.model_used} Demand (MW)`,
    line: { color: '#06b6d4', width: 3 },
    marker: {
      size: 6,
      color: forecast.map(f => f.status_color)
    }
  });

  const layout = {
    ...plotlyTheme,
    title: false,
    yaxis: { ...plotlyTheme.yaxis, title: 'Predicted Demand (MW)' },
    xaxis: { ...plotlyTheme.xaxis, title: 'Forecast Horizon (IST)' },
    legend: { orientation: 'h', y: 1.1, font: { color: '#cbd5e1' } }
  };

  Plotly.newPlot('forecastMainChart', traces, layout, { responsive: true, displayModeBar: false });

  // Filter Table
  filterForecastTable(document.getElementById('forecastTableFilter').value);
}

function filterForecastTable(filter) {
  if (!cachedForecastData) return;
  const tbody = document.getElementById('forecastTableBody');
  tbody.innerHTML = '';

  const forecast = cachedForecastData.forecast;
  const filtered = forecast.filter(item => {
    if (filter === 'ALL') return true;
    return item.status.toUpperCase() === filter;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--text-dim); padding:24px;">No hourly forecasts match the selected tier filter.</td></tr>`;
    return;
  }

  filtered.forEach(item => {
    const tr = document.createElement('tr');
    let recommendation = 'Maintain baseload generation.';
    if (item.status === 'Critical') recommendation = 'Dispatch fast-ramping peaker generation; initiate industrial curtailment.';
    else if (item.status === 'High') recommendation = 'Arm secondary spinning reserves; monitor inter-state ties.';
    else if (item.status === 'Low') recommendation = 'Demand valley; initiate battery / pumped-storage charging.';

    tr.innerHTML = `
      <td><strong>+${item.hour_num}h</strong></td>
      <td>${item.display_time}</td>
      <td style="font-weight:700; color:white;">${item.predicted_demand.toLocaleString()} MW</td>
      <td style="color:#94a3b8; font-size:0.83rem;">${item.lower_bound.toLocaleString()} - ${item.upper_bound.toLocaleString()} MW</td>
      <td><span class="badge-status ${item.status.toLowerCase()}">${item.status}</span></td>
      <td style="color:var(--text-muted); font-size:0.84rem;">${recommendation}</td>
    `;
    tbody.appendChild(tr);
  });
}

// 7. Fetch & Render Peak Load Analysis
async function fetchAndRenderPeakLoad() {
  showLoading('peakChartLoading', true);
  const model = document.getElementById('forecastModelSelect') ? document.getElementById('forecastModelSelect').value : 'LSTM';
  const crit = document.getElementById('threshCriticalSelect').value;
  const high = document.getElementById('threshHighSelect').value;

  try {
    const res = await fetch(`/api/peak-analysis?model=${encodeURIComponent(model)}&crit_thresh=${crit}&high_thresh=${high}`);
    cachedPeakData = await res.json();
    renderPeakLoadTab();
  } catch (err) {
    console.error('Peak load fetch failed:', err);
    showToast('Failed to compute peak load tiers.', 'error');
  } finally {
    showLoading('peakChartLoading', false);
  }
}

function renderPeakLoadTab() {
  if (!cachedPeakData) return;

  const summary = cachedPeakData.summary;
  document.getElementById('peakDemandValue').textContent = Number(summary.expected_peak_demand).toLocaleString();
  document.getElementById('peakTimeValue').textContent = summary.expected_peak_hour;
  document.getElementById('peakTimeDetails').textContent = `Occurs at ${summary.expected_peak_time}`;

  document.getElementById('peakAvgDemandValue').textContent = Number(summary.next_24h_avg).toLocaleString();
  document.getElementById('peakMinDemandValue').textContent = Number(summary.expected_min_demand).toLocaleString();
  document.getElementById('peakMinTimeDetails').textContent = `Occurs at ${summary.expected_min_time}`;

  document.getElementById('criticalCount').textContent = cachedPeakData.breakdown.critical_count;
  document.getElementById('highCount').textContent = cachedPeakData.breakdown.high_count;
  document.getElementById('normalCount').textContent = cachedPeakData.breakdown.normal_count;
  document.getElementById('lowCount').textContent = cachedPeakData.breakdown.low_count;

  // Bar chart of 24h stress tiers
  const times = cachedPeakData.hourly_patterns.map(h => h.display_time);
  const demands = cachedPeakData.hourly_patterns.map(h => h.predicted_demand);
  const colors = cachedPeakData.hourly_patterns.map(h => h.status_color);

  const barTrace = {
    x: times,
    y: demands,
    type: 'bar',
    marker: { color: colors, opacity: 0.9 },
    name: 'Predicted Load (MW)'
  };

  const layout = {
    ...plotlyTheme,
    yaxis: { ...plotlyTheme.yaxis, title: 'Demand (MW)' },
    xaxis: { ...plotlyTheme.xaxis, title: 'Next 24 Hours (Asia/Kolkata)' }
  };

  Plotly.newPlot('peakHourlyBarChart', [barTrace], layout, { responsive: true, displayModeBar: false });

  // Alerts Box
  const alertsBox = document.getElementById('peakAlertsList');
  alertsBox.innerHTML = '';
  cachedPeakData.alerts.forEach(a => {
    const div = document.createElement('div');
    div.className = `alert-item ${a.level}`;
    div.innerHTML = `
      <i data-lucide="${a.level === 'critical' ? 'alert-triangle' : (a.level === 'warning' ? 'alert-circle' : 'info')}" style="width:20px;height:20px;flex-shrink:0;"></i>
      <div>
        <div style="font-weight:700; margin-bottom:2px;">${a.title}</div>
        <div style="font-size:0.82rem; opacity:0.9;">${a.message}</div>
      </div>
    `;
    alertsBox.appendChild(div);
  });

  if (window.lucide) lucide.createIcons();
}

// 8. Render Model Comparison
function renderModelComparisonTab() {
  if (!cachedModelData) return;

  const metrics = cachedModelData.metrics;
  const best = cachedModelData.best_model;

  document.getElementById('bestModelTitle').textContent = `Top Algorithm: ${best}`;
  document.getElementById('bestModelDesc').textContent =
    `Objectively selected with lowest Mean Absolute Error (${metrics[best].MAE} MW) and superior R² (${metrics[best].R2}) on holdout validation.`;

  const tbody = document.getElementById('modelMetricsTableBody');
  tbody.innerHTML = '';

  const models = Object.keys(metrics);
  models.forEach((m, idx) => {
    const isBest = m === best;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${m}</strong> ${isBest ? '<i data-lucide="award" style="width:14px;color:#fbbf24;display:inline-block;vertical-align:middle;"></i>' : ''}</td>
      <td style="color:${isBest ? '#34d399' : '#e2e8f0'}; font-weight:${isBest ? '700' : '400'};">${metrics[m].MAE} MW</td>
      <td style="color:${isBest ? '#34d399' : '#e2e8f0'}; font-weight:${isBest ? '700' : '400'};">${metrics[m].RMSE} MW</td>
      <td>${metrics[m].MAPE}%</td>
      <td style="font-weight:700; color:${isBest ? 'var(--cyan-light)' : '#e2e8f0'};">${metrics[m].R2}</td>
      <td>
        <span class="badge-status ${isBest ? 'normal' : 'low'}">
          ${isBest ? '★ Rank 1 (Active)' : `Rank ${idx + 1}`}
        </span>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // MAE / RMSE Comparison
  const traceMAE = {
    x: models,
    y: models.map(m => metrics[m].MAE),
    name: 'MAE (MW)',
    type: 'bar',
    marker: { color: '#06b6d4' }
  };
  const traceRMSE = {
    x: models,
    y: models.map(m => metrics[m].RMSE),
    name: 'RMSE (MW)',
    type: 'bar',
    marker: { color: '#3b82f6' }
  };

  Plotly.newPlot('modelErrorBarChart', [traceMAE, traceRMSE], {
    ...plotlyTheme,
    barmode: 'group',
    legend: { orientation: 'h', y: 1.15 }
  }, { responsive: true, displayModeBar: false });

  // R2 Comparison
  const traceR2 = {
    x: models,
    y: models.map(m => metrics[m].R2),
    type: 'bar',
    marker: { color: ['#60a5fa', '#34d399', '#06b6d4'] }
  };

  Plotly.newPlot('modelR2BarChart', [traceR2], {
    ...plotlyTheme,
    yaxis: { ...plotlyTheme.yaxis, title: 'R² Score', range: [0.92, 1.0] }
  }, { responsive: true, displayModeBar: false });

  // Holdout Residual Timeline
  const testEval = cachedModelData.test_eval;
  if (testEval) {
    const traceActual = {
      x: testEval.timestamps,
      y: testEval.actual,
      name: 'Actual Holdout Demand',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#ffffff', width: 2 }
    };
    const traceLR = {
      x: testEval.timestamps,
      y: testEval.lr_pred,
      name: 'Linear Regression',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#f59e0b', width: 1.5, dash: 'dot' }
    };
    const traceRF = {
      x: testEval.timestamps,
      y: testEval.rf_pred,
      name: 'Random Forest',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#10b981', width: 1.5 }
    };
    const traceLSTM = {
      x: testEval.timestamps,
      y: testEval.lstm_pred,
      name: 'LSTM Sequence',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#06b6d4', width: 2 }
    };

    Plotly.newPlot('modelComparisonTimelineChart', [traceActual, traceLR, traceRF, traceLSTM], {
      ...plotlyTheme,
      legend: { orientation: 'h', y: 1.15 },
      xaxis: { ...plotlyTheme.xaxis, title: 'Validation Period (IST)' }
    }, { responsive: true, displayModeBar: false });
  }

  if (window.lucide) lucide.createIcons();
}

// 9. Render Dataset Management & Quality Audit
function renderDatasetTab() {
  if (!cachedDatasetData) return;

  const data = cachedDatasetData;
  const qr = data.quality_report || {};

  document.getElementById('qrTotalRows').textContent = Number(qr.total_rows || data.total_rows).toLocaleString();
  document.getElementById('qrDateRange').textContent = `Coverage: ${qr.date_range || 'N/A'}`;
  document.getElementById('qrCompleteness').textContent = `${qr.data_completeness_pct || 100}%`;
  document.getElementById('qrMissingValues').textContent = Number(qr.total_missing || 0).toLocaleString();
  document.getElementById('qrDuplicates').textContent = Number(qr.duplicate_timestamps || 0).toLocaleString();
  document.getElementById('qrMissingGaps').textContent = Number(qr.missing_hourly_intervals || 0).toLocaleString();

  // Preview Headers & Rows
  const thead = document.getElementById('previewTableHead');
  thead.innerHTML = '';
  data.columns.forEach(c => {
    const th = document.createElement('th');
    th.textContent = c;
    thead.appendChild(th);
  });

  const tbody = document.getElementById('previewTableBody');
  tbody.innerHTML = '';
  data.preview.forEach(row => {
    const tr = document.createElement('tr');
    data.columns.forEach(c => {
      const td = document.createElement('td');
      td.textContent = row[c] !== undefined ? row[c] : '--';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

// 10. CSV Upload Handler
async function handleCsvUpload(file) {
  const statusMsg = document.getElementById('uploadStatusMsg');
  const overlay = document.getElementById('uploadLoadingOverlay');
  overlay.classList.add('active');
  statusMsg.style.color = 'var(--cyan-light)';
  statusMsg.textContent = `Uploading and parsing ${file.name}...`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload-dataset', {
      method: 'POST',
      body: formData
    });
    const result = await res.json();

    if (result.success) {
      statusMsg.style.color = '#34d399';
      statusMsg.textContent = result.message;
      showToast('Dataset validated and models successfully retrained!');
      loadAllData(true);
    } else {
      statusMsg.style.color = '#f87171';
      statusMsg.textContent = result.error || 'Upload rejected.';
      showToast(result.error, 'error');
    }
  } catch (err) {
    statusMsg.style.color = '#f87171';
    statusMsg.textContent = 'Upload failed due to connection error.';
    showToast('Failed to upload file.', 'error');
  } finally {
    overlay.classList.remove('active');
  }
}

// 11. Reset to Default Dataset
async function resetToDefaultDataset() {
  const statusMsg = document.getElementById('uploadStatusMsg');
  statusMsg.style.color = 'var(--cyan-light)';
  statusMsg.textContent = 'Resetting to demonstration dataset...';

  try {
    const res = await fetch('/api/reset-dataset', { method: 'POST' });
    const result = await res.json();
    if (result.success) {
      statusMsg.style.color = '#34d399';
      statusMsg.textContent = result.message;
      showToast('Reverted to dynamic demonstration dataset.');
      loadAllData(true);
    }
  } catch (err) {
    statusMsg.style.color = '#f87171';
    statusMsg.textContent = 'Failed to reset dataset.';
  }
}

// Helpers
function showLoading(overlayId, show) {
  const el = document.getElementById(overlayId);
  if (el) {
    if (show) el.classList.add('active');
    else el.classList.remove('active');
  }
}

function showToast(message, type = 'success') {
  // Simple toast alert
  const toast = document.createElement('div');
  toast.style.position = 'fixed';
  toast.style.bottom = '24px';
  toast.style.right = '24px';
  toast.style.background = type === 'error' ? 'rgba(239, 68, 68, 0.95)' : 'rgba(6, 182, 212, 0.95)';
  toast.style.color = 'white';
  toast.style.padding = '12px 20px';
  toast.style.borderRadius = '8px';
  toast.style.boxShadow = '0 4px 20px rgba(0,0,0,0.5)';
  toast.style.zIndex = '9999';
  toast.style.fontSize = '0.88rem';
  toast.style.fontWeight = '500';
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3500);
}
