/**
 * analytics.js — All Chart.js visualizations for the Analytics page
 * Student Performance Prediction System | DAV Mini Project
 *
 * Charts rendered:
 *   1. Performance Distribution (Doughnut)
 *   2. Algorithm Accuracy Comparison (Bar)
 *   3. Study Hours vs Performance (Grouped Bar — mean)
 *   4. Absences vs Performance (Horizontal Bar)
 *   5. Study Time vs GPA Scatter
 *   6. Feature Importance (Horizontal Bar)
 *   7. Confusion Matrices (HTML table — DT & RF)
 *   8. Metrics Comparison Radar
 *   9. Model Comparison Table
 */

// ── Chart.js global defaults ───────────────────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.07)';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";

const COLORS = {
  low:    '#ef4444',
  medium: '#f59e0b',
  high:   '#22c55e',
  dt:     '#3b82f6',
  rf:     '#22c55e',
  indigo: '#6366f1',
  purple: '#a855f7',
  teal:   '#14b8a6',
};

// ── Wait for DOM ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  fetch('/model-metrics')
    .then(r => r.json())
    .then(data => {
      renderPerformanceDist(data);
      renderAccuracyComp(data);
      renderStudyHours(data);
      renderAbsences(data);
      renderScatter(data);
      renderFeatureImportance(data);
      renderConfusionMatrices(data);
      renderMetricsRadar(data);
      renderComparisonTable(data);
    })
    .catch(err => console.error('Failed to load metrics:', err));
});

// ─────────────────────────────────────────────────────────────
// 1. PERFORMANCE DISTRIBUTION — Doughnut
// ─────────────────────────────────────────────────────────────
function renderPerformanceDist(data) {
  const dist  = data.dataset.performance_distribution;
  const labels = ['Low', 'Medium', 'High'];
  const values = labels.map(l => dist[l] || 0);

  new Chart(document.getElementById('chartPerformanceDist'), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: [COLORS.low, COLORS.medium, COLORS.high],
        borderColor: '#131929',
        borderWidth: 3,
        hoverOffset: 10
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '62%',
      plugins: {
        legend: { position: 'bottom', labels: { padding: 20, font: { size: 12 } } },
        tooltip: {
          callbacks: {
            label: ctx => {
              const total = ctx.dataset.data.reduce((a,b) => a+b, 0);
              const pct   = ((ctx.parsed / total) * 100).toFixed(1);
              return ` ${ctx.label}: ${ctx.parsed.toLocaleString()} (${pct}%)`;
            }
          }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────
// 2. ALGORITHM ACCURACY COMPARISON — Bar
// ─────────────────────────────────────────────────────────────
function renderAccuracyComp(data) {
  const metrics = ['Accuracy','Precision','Recall','F1 Score'];
  const dtVals  = [
    data.decision_tree.accuracy,
    data.decision_tree.precision,
    data.decision_tree.recall,
    data.decision_tree.f1_score
  ].map(v => parseFloat((v * 100).toFixed(2)));
  const rfVals  = [
    data.random_forest.accuracy,
    data.random_forest.precision,
    data.random_forest.recall,
    data.random_forest.f1_score
  ].map(v => parseFloat((v * 100).toFixed(2)));

  new Chart(document.getElementById('chartAccuracyComp'), {
    type: 'bar',
    data: {
      labels: metrics,
      datasets: [
        {
          label: 'Decision Tree',
          data: dtVals,
          backgroundColor: 'rgba(59,130,246,0.7)',
          borderColor: COLORS.dt,
          borderWidth: 1.5,
          borderRadius: 5
        },
        {
          label: 'Random Forest',
          data: rfVals,
          backgroundColor: 'rgba(34,197,94,0.7)',
          borderColor: COLORS.rf,
          borderWidth: 1.5,
          borderRadius: 5
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' },
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%` } }
      },
      scales: {
        y: {
          min: 0, max: 100,
          ticks: { callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        x: { grid: { display: false } }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────
// 3. STUDY HOURS VS PERFORMANCE — Grouped Bar (mean + quartiles)
// ─────────────────────────────────────────────────────────────
function renderStudyHours(data) {
  const viz    = data.visualizations.study_hours_by_performance;
  const levels = ['Low', 'Medium', 'High'];
  const means  = levels.map(l => viz[l] ? viz[l].mean : 0);
  const colors = [COLORS.low, COLORS.medium, COLORS.high];

  new Chart(document.getElementById('chartStudyHours'), {
    type: 'bar',
    data: {
      labels: levels,
      datasets: [{
        label: 'Avg. Study Hours / Week',
        data: means,
        backgroundColor: colors.map(c => c + 'b0'),
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: ctx => {
              const lvl = levels[ctx.dataIndex];
              const d   = viz[lvl];
              return `  Q1: ${d.q1}h  |  Median: ${d.median}h  |  Q3: ${d.q3}h`;
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: 'Hours / Week', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        x: { grid: { display: false } }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────
// 4. ABSENCES VS PERFORMANCE — Horizontal Bar
// ─────────────────────────────────────────────────────────────
function renderAbsences(data) {
  const viz    = data.visualizations.absences_by_performance;
  const levels = ['Low', 'Medium', 'High'];
  const values = levels.map(l => viz[l] || 0);
  const colors = [COLORS.low, COLORS.medium, COLORS.high];

  new Chart(document.getElementById('chartAbsences'), {
    type: 'bar',
    data: {
      labels: levels,
      datasets: [{
        label: 'Avg. Absences',
        data: values,
        backgroundColor: colors.map(c => c + 'b0'),
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` Avg. Absences: ${ctx.parsed.x.toFixed(1)}` } }
      },
      scales: {
        x: {
          beginAtZero: true,
          title: { display: true, text: 'Average Absences', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: { grid: { display: false } }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────
// 5. STUDY TIME VS GPA — Scatter Plot
// ─────────────────────────────────────────────────────────────
function renderScatter(data) {
  const pts   = data.visualizations.scatter_study_gpa;
  const byLvl = { Low: [], Medium: [], High: [] };
  pts.forEach(p => { if (byLvl[p.level]) byLvl[p.level].push({ x: p.x, y: p.y }); });

  new Chart(document.getElementById('chartScatter'), {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Low',    data: byLvl.Low,    backgroundColor: COLORS.low    + 'bb', pointRadius: 5 },
        { label: 'Medium', data: byLvl.Medium, backgroundColor: COLORS.medium + 'bb', pointRadius: 5 },
        { label: 'High',   data: byLvl.High,   backgroundColor: COLORS.high   + 'bb', pointRadius: 5 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' },
        tooltip: { callbacks: { label: ctx => ` Study: ${ctx.parsed.x}h/wk | GPA: ${ctx.parsed.y}` } }
      },
      scales: {
        x: {
          title: { display: true, text: 'Study Time (hrs/week)', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          title: { display: true, text: 'GPA (0 – 4.0)', color: '#94a3b8' },
          min: 0, max: 4,
          grid: { color: 'rgba(255,255,255,0.05)' }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────
// 6. FEATURE IMPORTANCE — Horizontal Bar (Random Forest)
// ─────────────────────────────────────────────────────────────
function renderFeatureImportance(data) {
  const fi     = data.random_forest.feature_importances;
  const labels = Object.keys(fi);
  const values = Object.values(fi).map(v => parseFloat((v * 100).toFixed(2)));

  // Color gradient based on rank
  const palette = ['#6366f1','#818cf8','#a5b4fc','#3b82f6','#60a5fa','#93c5fd',
                   '#14b8a6','#2dd4bf','#5eead4','#22c55e','#4ade80','#86efac'];

  new Chart(document.getElementById('chartFeatureImportance'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance (%)',
        data: values,
        backgroundColor: palette.slice(0, labels.length),
        borderWidth: 0,
        borderRadius: 5
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` Importance: ${ctx.parsed.x.toFixed(2)}%` } }
      },
      scales: {
        x: {
          beginAtZero: true,
          ticks: { callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: { grid: { display: false } }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────
// 7. CONFUSION MATRICES — HTML tables
// ─────────────────────────────────────────────────────────────
function renderConfusionMatrices(data) {
  const labels = ['Low', 'Medium', 'High'];
  buildCM('cmDT', data.decision_tree.confusion_matrix,  labels);
  buildCM('cmRF', data.random_forest.confusion_matrix, labels);
}

function buildCM(containerId, cm, labels) {
  const container = document.getElementById(containerId);
  if (!container) return;

  let html = '<table class="cm-table"><thead><tr><th rowspan="2" style="text-align:left">Actual \\ Predicted</th>';
  labels.forEach(l => { html += `<th>${l}</th>`; });
  html += '</tr></thead><tbody>';

  const rowTotals = cm.map(row => row.reduce((a,b) => a+b, 0));

  cm.forEach((row, i) => {
    html += `<tr><th>${labels[i]}</th>`;
    row.forEach((cell, j) => {
      const isCorrect = i === j;
      const total     = rowTotals[i];
      const pct       = total > 0 ? ((cell / total) * 100).toFixed(1) : '0.0';
      const cls       = isCorrect ? 'cm-cell-correct' : (cell > 0 ? 'cm-cell-error' : '');
      html += `<td class="${cls}">${cell}<br/><small style="opacity:0.6;font-size:0.7rem">${pct}%</small></td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

// ─────────────────────────────────────────────────────────────
// 8. METRICS RADAR — Comparison
// ─────────────────────────────────────────────────────────────
function renderMetricsRadar(data) {
  const metrics = ['Accuracy','Precision','Recall','F1 Score'];
  const dtVals  = [
    data.decision_tree.accuracy,
    data.decision_tree.precision,
    data.decision_tree.recall,
    data.decision_tree.f1_score
  ].map(v => parseFloat((v * 100).toFixed(2)));
  const rfVals  = [
    data.random_forest.accuracy,
    data.random_forest.precision,
    data.random_forest.recall,
    data.random_forest.f1_score
  ].map(v => parseFloat((v * 100).toFixed(2)));

  new Chart(document.getElementById('chartMetricsComp'), {
    type: 'radar',
    data: {
      labels: metrics,
      datasets: [
        {
          label: 'Decision Tree',
          data: dtVals,
          backgroundColor: 'rgba(59,130,246,0.15)',
          borderColor: COLORS.dt,
          pointBackgroundColor: COLORS.dt,
          borderWidth: 2
        },
        {
          label: 'Random Forest',
          data: rfVals,
          backgroundColor: 'rgba(34,197,94,0.15)',
          borderColor: COLORS.rf,
          pointBackgroundColor: COLORS.rf,
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'top' } },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 25, callback: v => v + '%', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.08)' },
          angleLines: { color: 'rgba(255,255,255,0.08)' },
          pointLabels: { font: { size: 11 }, color: '#94a3b8' }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────
// 9. MODEL COMPARISON TABLE
// ─────────────────────────────────────────────────────────────
function renderComparisonTable(data) {
  const metrics = [
    { label: 'Accuracy',  dt: data.decision_tree.accuracy,  rf: data.random_forest.accuracy  },
    { label: 'Precision', dt: data.decision_tree.precision, rf: data.random_forest.precision },
    { label: 'Recall',    dt: data.decision_tree.recall,    rf: data.random_forest.recall    },
    { label: 'F1 Score',  dt: data.decision_tree.f1_score,  rf: data.random_forest.f1_score  }
  ];

  const tbody = document.getElementById('metricsTableBody');
  if (!tbody) return;

  let rows = '';
  metrics.forEach(m => {
    const dtPct  = (m.dt * 100).toFixed(2) + '%';
    const rfPct  = (m.rf * 100).toFixed(2) + '%';
    const diff   = ((m.rf - m.dt) * 100).toFixed(2);
    const diffStr = diff > 0
      ? `<span style="color:#22c55e">+${diff}%</span>`
      : diff < 0
        ? `<span style="color:#ef4444">${diff}%</span>`
        : `<span style="color:#94a3b8">0.00%</span>`;

    rows += `<tr>
      <td>${m.label}</td>
      <td style="font-family:var(--font-mono);font-weight:600;color:#60a5fa">${dtPct}</td>
      <td style="font-family:var(--font-mono);font-weight:600;color:#4ade80">${rfPct}</td>
      <td>${diffStr}</td>
    </tr>`;
  });
  tbody.innerHTML = rows;

  // Observation
  const obs = document.getElementById('modelObservation');
  if (!obs) return;

  const rfBetter = data.random_forest.accuracy >= data.decision_tree.accuracy;
  const higher   = rfBetter ? 'Random Forest' : 'Decision Tree';
  const lower    = rfBetter ? 'Decision Tree' : 'Random Forest';
  const diffAcc  = Math.abs((data.random_forest.accuracy - data.decision_tree.accuracy) * 100).toFixed(2);

  obs.innerHTML = `
    <strong>Observation:</strong> On this dataset, <strong>${higher}</strong> achieves a higher accuracy
    than <strong>${lower}</strong> by <strong>${diffAcc}%</strong>.
    ${rfBetter
      ? 'This is expected — Random Forest reduces overfitting by averaging predictions from 200 trees, leading to better generalization.'
      : 'This can occur when the dataset has clear, simple decision boundaries that a single Decision Tree captures effectively.'}
    Both models were trained on 80% of the data and evaluated on the unseen 20% test set.
  `;
  obs.style.display = 'block';
}
