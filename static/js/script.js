/**
 * script.js — Shared utilities for Student Performance Prediction System
 * DAV Mini Project
 */

// ── Mobile nav toggle ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('navToggle');
  const links  = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', () => links.classList.toggle('open'));
  }
});

// ── Populate stat cards from metrics JSON ──────────────────────
function populateStatCards(data) {
  document.querySelectorAll('.stat-card[data-metric]').forEach(card => {
    const metricPath = card.dataset.metric;  // e.g. "dataset.total_students"
    const label      = card.dataset.label;
    const icon       = card.dataset.icon;
    const isPct      = card.dataset.pct === 'true';

    // Walk the data object using the dot-path
    const value = metricPath.split('.').reduce((obj, key) => (obj || {})[key], data);

    let displayVal = value;
    if (value !== undefined && value !== null) {
      if (isPct) {
        displayVal = (parseFloat(value) * 100).toFixed(1) + '%';
      } else if (typeof value === 'number' && !isPct) {
        displayVal = value.toLocaleString();
      }
    } else {
      displayVal = '—';
    }

    card.innerHTML = `
      <div class="stat-icon">${icon}</div>
      <div class="stat-value">${displayVal}</div>
      <div class="stat-label">${label}</div>
    `;
  });
}

// ── Number counter animation ───────────────────────────────────
function animateCount(element, target, duration = 1200) {
  const start    = 0;
  const startTs  = performance.now();
  const isFloat  = String(target).includes('.');
  const decimals = isFloat ? 2 : 0;

  function step(ts) {
    const elapsed  = ts - startTs;
    const progress = Math.min(elapsed / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);
    const current  = start + eased * (target - start);
    element.textContent = isFloat ? current.toFixed(decimals) : Math.round(current).toLocaleString();
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
