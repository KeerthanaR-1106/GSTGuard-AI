/**
 * dashboard.js
 * GST Invoice Fraud Detection System - Dashboard Page Script
 *
 * Reads summary data from:
 *   1. window.dashboardSummary (set inline by Jinja2 template), OR
 *   2. <div id="dashboard-data" data-summary='{"..."}'>
 *
 * Initialises Chart.js charts and updates DOM stat cards.
 */

'use strict';

/* ============================================================
   Utility: Format number with comma separators
   ============================================================ */
/**
 * formatNumber - adds thousands comma separators.
 * @param {number|string} n
 * @returns {string}
 */
function formatNumber(n) {
  var num = parseFloat(n);
  if (isNaN(num)) return String(n);
  if (Number.isInteger(num)) {
    return num.toLocaleString('en-IN');
  }
  return num.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/* ============================================================
   Utility: Colour resolver for risk scores
   ============================================================ */
/**
 * riskScoreColor - returns CSS colour string for a 0-100 risk score.
 * @param {number} score
 * @returns {string}
 */
function riskScoreColor(score) {
  if (score >= 70) return '#dc3545';  /* High  */
  if (score >= 40) return '#ffc107';  /* Medium */
  return '#28a745';                   /* Low   */
}

/* ============================================================
   Risk score progress bars
   ============================================================ */
/**
 * initRiskProgressBars - Reads all [data-risk-score] progress wrappers
 * and sets width + colour of the inner .progress-bar.
 */
function initRiskProgressBars() {
  var wrappers = document.querySelectorAll('.risk-progress[data-score]');
  wrappers.forEach(function (wrapper) {
    var score = parseFloat(wrapper.getAttribute('data-score')) || 0;
    var bar   = wrapper.querySelector('.progress-bar');
    if (!bar) return;
    bar.style.width = Math.min(score, 100) + '%';
    bar.style.backgroundColor = riskScoreColor(score);
    bar.setAttribute('aria-valuenow', score);
  });
}

/* ============================================================
   Update DOM stat cards from summary object
   ============================================================ */
/**
 * updateStatCards - pushes summary values into DOM elements by id.
 * All element IDs are optional; missing ones are silently skipped.
 *
 * Expected keys in summary:
 *   total_invoices, high_risk_count, medium_risk_count, low_risk_count,
 *   flagged_count, avg_risk_score, total_amount, flagged_amount
 *
 * @param {Object} summary
 */
function updateStatCards(summary) {
  if (!summary || typeof summary !== 'object') return;

  var mappings = {
    'stat-total-invoices'  : formatNumber(summary.total_invoices   || 0),
    'stat-high-risk'       : formatNumber(summary.high_risk_count  || 0),
    'stat-medium-risk'     : formatNumber(summary.medium_risk_count|| 0),
    'stat-low-risk'        : formatNumber(summary.low_risk_count   || 0),
    'stat-flagged'         : formatNumber(summary.flagged_count    || 0),
    'stat-avg-risk-score'  : (parseFloat(summary.avg_risk_score)   || 0).toFixed(1),
    'stat-total-amount'    : formatNumber(summary.total_amount     || 0),
    'stat-flagged-amount'  : formatNumber(summary.flagged_amount   || 0)
  };

  Object.keys(mappings).forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.textContent = mappings[id];
  });

  /* Optional percentage badges */
  if (summary.total_invoices && summary.high_risk_count !== undefined) {
    var pctEl = document.getElementById('stat-high-risk-pct');
    if (pctEl) {
      var pct = ((summary.high_risk_count / summary.total_invoices) * 100).toFixed(1);
      pctEl.textContent = pct + '% of total';
    }
  }
}

/* ============================================================
   Chart 1: Doughnut - Risk Level Distribution
   ============================================================ */
/**
 * initRiskChart - builds or rebuilds the risk doughnut chart.
 * @param {Object} summary  - must have low_risk_count, medium_risk_count, high_risk_count
 * @returns {Chart|null}
 */
function initRiskChart(summary) {
  if (typeof Chart === 'undefined') {
    console.warn('[dashboard.js] Chart.js is not loaded. Skipping riskChart.');
    return null;
  }

  var canvas = document.getElementById('riskChart');
  if (!canvas) return null;

  var low    = parseInt(summary.low_risk_count,    10) || 0;
  var medium = parseInt(summary.medium_risk_count, 10) || 0;
  var high   = parseInt(summary.high_risk_count,   10) || 0;
  var total  = low + medium + high || 1;  /* avoid division by zero */

  /* Destroy any previous chart instance on this canvas */
  var existing = Chart.getChart(canvas);
  if (existing) existing.destroy();

  return new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: ['Low Risk', 'Medium Risk', 'High Risk'],
      datasets: [{
        data: [low, medium, high],
        backgroundColor: ['#28a745', '#ffc107', '#dc3545'],
        hoverBackgroundColor: ['#218838', '#e0a800', '#c82333'],
        borderColor: '#ffffff',
        borderWidth: 3,
        hoverOffset: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            padding: 20,
            usePointStyle: true,
            pointStyleWidth: 10,
            font: { size: 12, family: "'Segoe UI', system-ui, sans-serif" }
          }
        },
        title: {
          display: true,
          text: 'Risk Level Distribution',
          font: { size: 14, weight: '600', family: "'Segoe UI', system-ui, sans-serif" },
          color: '#1e1e2e',
          padding: { bottom: 16 }
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              var label = context.label || '';
              var value = context.parsed || 0;
              var pct   = ((value / total) * 100).toFixed(1);
              return '  ' + label + ': ' + formatNumber(value) + ' (' + pct + '%)';
            }
          },
          backgroundColor: 'rgba(30,30,46,0.9)',
          titleFont: { size: 12 },
          bodyFont: { size: 12 },
          padding: 10,
          cornerRadius: 8
        }
      },
      animation: {
        animateScale: true,
        animateRotate: true,
        duration: 700,
        easing: 'easeInOutQuart'
      }
    }
  });
}

/* ============================================================
   Chart 2: Horizontal Bar - Anomaly Type Breakdown
   ============================================================ */
/**
 * initAnomalyChart - builds or rebuilds the anomaly horizontal bar chart.
 * Only shows types where count > 0.
 * @param {Object} anomalyTypes  - dict: { "type_label": count, ... }
 * @returns {Chart|null}
 */
function initAnomalyChart(anomalyTypes) {
  if (typeof Chart === 'undefined') {
    console.warn('[dashboard.js] Chart.js is not loaded. Skipping anomalyChart.');
    return null;
  }

  var canvas = document.getElementById('anomalyChart');
  if (!canvas) return null;

  /* Filter types with count > 0 */
  var filtered = {};
  if (anomalyTypes && typeof anomalyTypes === 'object') {
    Object.keys(anomalyTypes).forEach(function (key) {
      var count = parseInt(anomalyTypes[key], 10) || 0;
      if (count > 0) filtered[key] = count;
    });
  }

  var labels = Object.keys(filtered);
  var values = labels.map(function (k) { return filtered[k]; });

  /* Destroy any previous chart instance */
  var existing = Chart.getChart(canvas);
  if (existing) existing.destroy();

  if (labels.length === 0) {
    /* Show empty state message on canvas */
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#6c757d';
    ctx.font = '14px Segoe UI, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No anomaly data available', canvas.width / 2, canvas.height / 2);
    return null;
  }

  /* Generate shades of indigo with slight variation per bar */
  var barColors = labels.map(function (_, i) {
    var alpha = 0.75 + (i % 3) * 0.08;
    return 'rgba(79, 70, 229, ' + alpha + ')';
  });

  return new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Suspicious Invoices',
        data: values,
        backgroundColor: barColors,
        hoverBackgroundColor: '#3730a3',
        borderColor: '#4f46e5',
        borderWidth: 1,
        borderRadius: 5,
        borderSkipped: false
      }]
    },
    options: {
      indexAxis: 'y',   /* Horizontal bar */
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        title: {
          display: true,
          text: 'Suspicious Invoices by Anomaly Type',
          font: { size: 14, weight: '600', family: "'Segoe UI', system-ui, sans-serif" },
          color: '#1e1e2e',
          padding: { bottom: 16 }
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              return '  Count: ' + formatNumber(context.parsed.x);
            }
          },
          backgroundColor: 'rgba(30,30,46,0.9)',
          titleFont: { size: 12 },
          bodyFont: { size: 12 },
          padding: 10,
          cornerRadius: 8
        }
      },
      scales: {
        x: {
          beginAtZero: true,
          grid: {
            color: 'rgba(0,0,0,0.05)'
          },
          ticks: {
            precision: 0,
            font: { size: 11 },
            callback: function (value) { return formatNumber(value); }
          },
          title: {
            display: true,
            text: 'Number of Invoices',
            font: { size: 11, weight: '600' },
            color: '#6c757d'
          }
        },
        y: {
          grid: { display: false },
          ticks: {
            font: { size: 11 },
            autoSkip: false
          }
        }
      },
      animation: {
        duration: 700,
        easing: 'easeInOutQuart'
      }
    }
  });
}

/* ============================================================
   Sidebar toggle (mobile)
   ============================================================ */
function initSidebarToggle() {
  var toggleBtn = document.getElementById('sidebarToggle');
  var sidebar   = document.getElementById('sidebar');
  var overlay   = document.getElementById('sidebarOverlay');

  if (!toggleBtn || !sidebar) return;

  toggleBtn.addEventListener('click', function () {
    sidebar.classList.toggle('sidebar-open');
    if (overlay) overlay.classList.toggle('active');
  });

  if (overlay) {
    overlay.addEventListener('click', function () {
      sidebar.classList.remove('sidebar-open');
      overlay.classList.remove('active');
    });
  }

  /* Close sidebar on Escape */
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
      sidebar.classList.remove('sidebar-open');
      if (overlay) overlay.classList.remove('active');
    }
  });
}

/* ============================================================
   Upload Dropzone enhancements
   ============================================================ */
function initDropzone() {
  var zones = document.querySelectorAll('.dropzone');
  zones.forEach(function (zone) {
    var input = zone.querySelector('input[type="file"]');

    zone.addEventListener('dragover', function (e) {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', function (e) {
      e.preventDefault();
      e.stopPropagation();
      /* Only remove if leaving the zone itself, not a child */
      if (!zone.contains(e.relatedTarget)) {
        zone.classList.remove('dragover');
      }
    });

    zone.addEventListener('drop', function (e) {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove('dragover');
      if (input && e.dataTransfer && e.dataTransfer.files.length > 0) {
        /* Transfer dropped files to the hidden input */
        try {
          input.files = e.dataTransfer.files;
        } catch (ex) {
          /* Some browsers don't allow direct assignment - silently ignore */
        }
        updateDropzoneLabel(zone, e.dataTransfer.files);
      }
    });

    if (input) {
      input.addEventListener('change', function () {
        if (input.files && input.files.length > 0) {
          updateDropzoneLabel(zone, input.files);
          zone.classList.add('file-selected');
        }
      });
    }
  });
}

function updateDropzoneLabel(zone, files) {
  var titleEl = zone.querySelector('.dropzone-title');
  var subEl   = zone.querySelector('.dropzone-sub');
  if (!titleEl) return;

  zone.classList.remove('dragover');
  zone.classList.add('file-selected');

  if (files.length === 1) {
    titleEl.textContent = files[0].name;
    if (subEl) {
      var size = (files[0].size / 1024).toFixed(1);
      if (files[0].size >= 1024 * 1024) {
        size = (files[0].size / (1024 * 1024)).toFixed(2) + ' MB';
      } else {
        size = size + ' KB';
      }
      subEl.textContent = size + ' - ready to upload';
    }
  } else {
    titleEl.textContent = files.length + ' files selected';
    if (subEl) subEl.textContent = 'Ready to upload';
  }
}

/* ============================================================
   Table: sort on column header click
   ============================================================ */
function initSortableTables() {
  var tables = document.querySelectorAll('table[data-sortable]');
  tables.forEach(function (table) {
    var headers = table.querySelectorAll('th.sortable');
    headers.forEach(function (th, colIndex) {
      th.addEventListener('click', function () {
        var tbody  = table.querySelector('tbody');
        if (!tbody) return;
        var rows   = Array.from(tbody.querySelectorAll('tr'));
        var asc    = th.classList.contains('asc');

        /* Reset all headers */
        headers.forEach(function (h) { h.classList.remove('asc', 'desc'); });

        rows.sort(function (a, b) {
          var aVal = (a.cells[colIndex] ? a.cells[colIndex].textContent.trim() : '');
          var bVal = (b.cells[colIndex] ? b.cells[colIndex].textContent.trim() : '');
          /* Try numeric comparison first */
          var aNum = parseFloat(aVal.replace(/,/g, ''));
          var bNum = parseFloat(bVal.replace(/,/g, ''));
          if (!isNaN(aNum) && !isNaN(bNum)) {
            return asc ? bNum - aNum : aNum - bNum;
          }
          return asc
            ? bVal.localeCompare(aVal)
            : aVal.localeCompare(bVal);
        });

        th.classList.add(asc ? 'desc' : 'asc');
        rows.forEach(function (row) { tbody.appendChild(row); });
      });
    });
  });
}

/* ============================================================
   Flash message auto-dismiss
   ============================================================ */
function initAlertAutoDismiss(delayMs) {
  var delay = delayMs || 5000;
  var alerts = document.querySelectorAll('.alert.alert-auto-dismiss');
  alerts.forEach(function (alert) {
    setTimeout(function () {
      alert.style.transition = 'opacity 0.5s ease';
      alert.style.opacity    = '0';
      setTimeout(function () { alert.remove(); }, 500);
    }, delay);
  });
}

/* ============================================================
   Read summary data from the page
   ============================================================ */
/**
 * getSummaryData - reads summary from window.dashboardSummary OR
 * from a <div id="dashboard-data" data-summary='...'> element.
 * @returns {Object|null}
 */
function getSummaryData() {
  /* Priority 1: Jinja2 inline window variable */
  if (window.dashboardSummary && typeof window.dashboardSummary === 'object') {
    return window.dashboardSummary;
  }

  /* Priority 2: data attribute on a DOM element */
  var dataEl = document.getElementById('dashboard-data');
  if (dataEl) {
    var raw = dataEl.getAttribute('data-summary');
    if (raw) {
      try {
        return JSON.parse(raw);
      } catch (e) {
        console.error('[dashboard.js] Failed to parse data-summary JSON:', e);
      }
    }
  }

  return null;
}

/* ============================================================
   Main Initialisation
   ============================================================ */
document.addEventListener('DOMContentLoaded', function () {

  /* --- Sidebar toggle (mobile) --- */
  initSidebarToggle();

  /* --- Dropzone --- */
  initDropzone();

  /* --- Sortable tables --- */
  initSortableTables();

  /* --- Risk progress bars --- */
  initRiskProgressBars();

  /* --- Auto-dismiss flash messages --- */
  initAlertAutoDismiss(5500);

  /* --- Read data and initialise charts / stat cards --- */
  var summary = getSummaryData();

  if (summary) {
    /* Update stat cards */
    updateStatCards(summary);

    /* Chart 1: Risk doughnut */
    initRiskChart(summary);

    /* Chart 2: Anomaly horizontal bar */
    var anomalyTypes = summary.anomaly_types || {};
    initAnomalyChart(anomalyTypes);
  } else {
    console.info('[dashboard.js] No dashboard summary data found. Charts will not render.');
  }

  /* --- Tooltip initialisation (Bootstrap 5) --- */
  if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
    var tooltipEls = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipEls.forEach(function (el) {
      new bootstrap.Tooltip(el, { trigger: 'hover' });
    });
  }

  /* --- Popover initialisation (Bootstrap 5) --- */
  if (typeof bootstrap !== 'undefined' && bootstrap.Popover) {
    var popoverEls = document.querySelectorAll('[data-bs-toggle="popover"]');
    popoverEls.forEach(function (el) {
      new bootstrap.Popover(el);
    });
  }

});

/* ============================================================
   Global exports - callable from inline Jinja2 <script> blocks
   ============================================================ */
window.Dashboard = {
  formatNumber       : formatNumber,
  riskScoreColor     : riskScoreColor,
  updateStatCards    : updateStatCards,
  initRiskChart      : initRiskChart,
  initAnomalyChart   : initAnomalyChart,
  initRiskProgressBars : initRiskProgressBars,
  getSummaryData     : getSummaryData
};
