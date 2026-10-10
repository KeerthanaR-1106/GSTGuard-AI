/**
 * static/js/charts.js
 * ====================
 * Chart.js helper utilities for GST Invoice Fraud Detection System.
 * Provides modular chart initializers for model comparisons and anomaly breakdowns.
 */

(function () {
  'use strict';

  /**
   * Initialize grouped bar chart comparing performance metrics across models.
   * @param {string} canvasId - The canvas element ID.
   * @param {Object} performanceData - Dictionary containing model metrics (accuracy, precision, recall, f1).
   */
  function initModelPerformanceChart(canvasId, performanceData) {
    var ctx = document.getElementById(canvasId);
    if (!ctx || typeof Chart === 'undefined') return null;

    var modelKeys = ['Isolation Forest', 'LOF', 'Z-Score'];
    var modelLabels = ['Isolation Forest', 'Local Outlier Factor (LOF)', 'Statistical Z-Score'];

    function getVal(key, field) {
      if (!performanceData || !performanceData[key]) return 0;
      var val = performanceData[key][field];
      return typeof val === 'number' ? parseFloat((val * 100).toFixed(2)) : 0;
    }

    var metricDefs = [
      { label: 'Accuracy', field: 'accuracy', color: 'rgba(54, 162, 235, 0.82)', border: 'rgba(54, 162, 235, 1)' },
      { label: 'Precision', field: 'precision', color: 'rgba(255, 159, 64, 0.82)', border: 'rgba(255, 159, 64, 1)' },
      { label: 'Recall', field: 'recall', color: 'rgba(75, 192, 192, 0.82)', border: 'rgba(75, 192, 192, 1)' },
      { label: 'F1-Score', field: 'f1', color: 'rgba(153, 102, 255, 0.82)', border: 'rgba(153, 102, 255, 1)' }
    ];

    var datasets = metricDefs.map(function (def) {
      return {
        label: def.label,
        data: modelKeys.map(function (k) { return getVal(k, def.field); }),
        backgroundColor: def.color,
        borderColor: def.border,
        borderWidth: 1.5,
        borderRadius: 4
      };
    });

    return new Chart(ctx, {
      type: 'bar',
      data: {
        labels: modelLabels,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { font: { size: 12 }, padding: 12 }
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                return ' ' + context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%';
              }
            }
          }
        },
        scales: {
          x: { grid: { display: false } },
          y: {
            beginAtZero: true,
            max: 100,
            ticks: {
              callback: function (val) { return val + '%'; }
            },
            grid: { color: 'rgba(0, 0, 0, 0.06)' }
          }
        }
      }
    });
  }

  /**
   * Initialize bar chart for per-type anomaly detection rates.
   * @param {string} canvasId - The canvas element ID.
   * @param {Object} perTypeData - Dictionary mapping anomaly types to detection stats.
   */
  function initAnomalyTypeChart(canvasId, perTypeData) {
    var ctx = document.getElementById(canvasId);
    if (!ctx || typeof Chart === 'undefined' || !perTypeData) return null;

    var labels = Object.keys(perTypeData);
    var rates = labels.map(function (label) {
      var item = perTypeData[label];
      return typeof item === 'number' ? item : (item && item.rate ? item.rate * 100 : 0);
    });

    return new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Detection Rate (%)',
          data: rates,
          backgroundColor: 'rgba(79, 70, 229, 0.8)',
          borderColor: 'rgba(79, 70, 229, 1)',
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ' ' + ctx.parsed.x.toFixed(1) + '% detected';
              }
            }
          }
        },
        scales: {
          x: {
            beginAtZero: true,
            max: 100,
            ticks: {
              callback: function (val) { return val + '%'; }
            }
          },
          y: {
            grid: { display: false }
          }
        }
      }
    });
  }

  // Export globally
  window.GSTCharts = {
    initModelPerformanceChart: initModelPerformanceChart,
    initAnomalyTypeChart: initAnomalyTypeChart
  };
})();
