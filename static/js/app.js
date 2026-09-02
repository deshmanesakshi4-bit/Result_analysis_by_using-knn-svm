/**
 * app.js
 * Frontend controller for EduPredict ML Dashboard.
 * Handles live REST API queries, real-time predictions, and dynamic Chart.js updates.
 */

let currentStrategy = 'mean';
let accuracyChart = null;
let metricsChart = null;
let predictDebounceTimer = null;

// Initialization on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadDashboard();
});

function loadDashboard() {
    fetchDataset();
    fetchModelComparison();
    fetchImputationComparison();
    runPrediction();
}

function onStrategyChange(newStrategy) {
    currentStrategy = newStrategy;
    loadDashboard();
}

/* ==========================================================================
   1. Dataset & Missing Value Inspection
   ========================================================================== */
async function fetchDataset() {
    try {
        const res = await fetch(`/api/data?strategy=${currentStrategy}`);
        const data = await res.json();

        // Update Imputed Value Pills
        document.getElementById('pill-imp-marks').innerText = data.imputed_values.Previous_Marks ?? '--';
        document.getElementById('pill-imp-hours').innerText = (data.imputed_values.Study_Hours ?? '--') + ' hrs';
        document.getElementById('pill-imp-att').innerText = (data.imputed_values.Attendance ?? '--') + '%';

        // Render Table Body
        const tbody = document.getElementById('students-tbody');
        tbody.innerHTML = '';

        data.raw_records.forEach((raw, i) => {
            const imp = data.imputed_records[i];
            const tr = document.createElement('tr');

            // Marks cell
            const marksCell = raw.Previous_Marks === null
                ? `<td><span class="tag-missing"><i class="fa-solid fa-triangle-exclamation"></i> Missing</span> &rarr; <span class="tag-imputed">${imp.Previous_Marks}</span></td>`
                : `<td>${raw.Previous_Marks}</td>`;

            // Hours cell
            const hoursCell = raw.Study_Hours === null
                ? `<td><span class="tag-missing"><i class="fa-solid fa-triangle-exclamation"></i> Missing</span> &rarr; <span class="tag-imputed">${imp.Study_Hours}</span></td>`
                : `<td>${raw.Study_Hours}</td>`;

            // Attendance cell
            const attCell = raw.Attendance === null
                ? `<td><span class="tag-missing"><i class="fa-solid fa-triangle-exclamation"></i> Missing</span> &rarr; <span class="tag-imputed">${imp.Attendance}%</span></td>`
                : `<td>${raw.Attendance}%</td>`;

            const resBadge = raw.Result.toLowerCase() === 'pass'
                ? `<span class="badge-pass"><i class="fa-solid fa-check"></i> Pass</span>`
                : `<span class="badge-fail"><i class="fa-solid fa-xmark"></i> Fail</span>`;

            const statusCell = raw.has_missing
                ? `<td><span class="badge-status">Imputed (${currentStrategy.toUpperCase()})</span></td>`
                : `<td><span style="color: var(--text-muted); font-size: 0.8rem;">Complete</span></td>`;

            tr.innerHTML = `
                <td><strong>#${raw.Student_ID}</strong></td>
                ${marksCell}
                ${hoursCell}
                ${attCell}
                <td>${resBadge}</td>
                ${statusCell}
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.error('Error fetching dataset:', err);
    }
}

/* ==========================================================================
   2. Model Comparison & Metrics Evaluation
   ========================================================================== */
async function fetchModelComparison() {
    try {
        const k = document.getElementById('knn-k-select').value;
        const kernel = document.getElementById('svm-kernel-select').value;

        const res = await fetch(`/api/models?strategy=${currentStrategy}&k=${k}&kernel=${kernel}`);
        const data = await res.json();

        // 1. Metric Cards
        document.getElementById('knn-test-acc-val').innerText = `${data.knn.test_accuracy}%`;
        document.getElementById('knn-cv-val').innerText = `${data.knn.cv_accuracy_mean}% (±${data.knn.cv_accuracy_std}%)`;

        document.getElementById('svm-test-acc-val').innerText = `${data.svm.test_accuracy}%`;
        document.getElementById('svm-cv-val').innerText = `${data.svm.cv_accuracy_mean}% (±${data.svm.cv_accuracy_std}%)`;

        document.getElementById('knn-prec-val').innerText = `${data.knn.precision}%`;
        document.getElementById('svm-prec-val').innerText = `${data.svm.precision}%`;

        document.getElementById('knn-f1-val').innerText = `${data.knn.f1_score}%`;
        document.getElementById('svm-f1-val').innerText = `${data.svm.f1_score}%`;

        // 2. Table Update
        document.getElementById('tbl-knn-param').innerText = `k=${k}`;
        document.getElementById('tbl-svm-param').innerText = kernel === 'linear' ? 'Linear' : 'RBF';

        document.getElementById('tbl-knn-train-acc').innerText = `${data.knn.train_accuracy}%`;
        document.getElementById('tbl-svm-train-acc').innerText = `${data.svm.train_accuracy}%`;
        updateCompBadge('tbl-train-comp', data.knn.train_accuracy, data.svm.train_accuracy);

        document.getElementById('tbl-knn-test-acc').innerText = `${data.knn.test_accuracy}%`;
        document.getElementById('tbl-svm-test-acc').innerText = `${data.svm.test_accuracy}%`;
        updateCompBadge('tbl-test-comp', data.knn.test_accuracy, data.svm.test_accuracy);

        document.getElementById('tbl-knn-cv-acc').innerText = `${data.knn.cv_accuracy_mean}%`;
        document.getElementById('tbl-svm-cv-acc').innerText = `${data.svm.cv_accuracy_mean}%`;
        updateCompBadge('tbl-cv-comp', data.knn.cv_accuracy_mean, data.svm.cv_accuracy_mean);

        document.getElementById('tbl-knn-prec').innerText = `${data.knn.precision}%`;
        document.getElementById('tbl-svm-prec').innerText = `${data.svm.precision}%`;
        updateCompBadge('tbl-prec-comp', data.knn.precision, data.svm.precision);

        document.getElementById('tbl-knn-rec').innerText = `${data.knn.recall}%`;
        document.getElementById('tbl-svm-rec').innerText = `${data.svm.recall}%`;
        updateCompBadge('tbl-rec-comp', data.knn.recall, data.svm.recall);

        document.getElementById('tbl-knn-f1').innerText = `${data.knn.f1_score}%`;
        document.getElementById('tbl-svm-f1').innerText = `${data.svm.f1_score}%`;
        updateCompBadge('tbl-f1-comp', data.knn.f1_score, data.svm.f1_score);

        // 3. Confusion Matrices
        document.getElementById('knn-cm-tn').innerText = data.knn.confusion_matrix[0][0];
        document.getElementById('knn-cm-fp').innerText = data.knn.confusion_matrix[0][1];
        document.getElementById('knn-cm-fn').innerText = data.knn.confusion_matrix[1][0];
        document.getElementById('knn-cm-tp').innerText = data.knn.confusion_matrix[1][1];

        document.getElementById('svm-cm-tn').innerText = data.svm.confusion_matrix[0][0];
        document.getElementById('svm-cm-fp').innerText = data.svm.confusion_matrix[0][1];
        document.getElementById('svm-cm-fn').innerText = data.svm.confusion_matrix[1][0];
        document.getElementById('svm-cm-tp').innerText = data.svm.confusion_matrix[1][1];

        // 4. Update Charts
        updateCharts(data);

        // 5. Update Predictor Model Labels
        document.getElementById('knn-meta-text').innerText = `k = ${k}`;
        document.getElementById('svm-meta-text').innerText = `${kernel === 'linear' ? 'Linear' : 'RBF'} Kernel`;

    } catch (err) {
        console.error('Error fetching model comparison:', err);
    }
}

function updateCompBadge(elementId, knnScore, svmScore) {
    const el = document.getElementById(elementId);
    const diff = svmScore - knnScore;
    if (Math.abs(diff) < 0.01) {
        el.innerHTML = `<span class="badge-status">Tied</span>`;
    } else if (diff > 0) {
        el.innerHTML = `<span class="badge-status" style="background: rgba(248, 113, 113, 0.2); color: var(--accent-red);">SVM (+${diff.toFixed(1)}%)</span>`;
    } else {
        el.innerHTML = `<span class="badge-status" style="background: rgba(56, 189, 248, 0.2); color: var(--accent-blue);">KNN (+${(-diff).toFixed(1)}%)</span>`;
    }
}

/* ==========================================================================
   3. Live Interactive Predictor
   ========================================================================== */
function syncInputs(type, val) {
    if (type === 'marks') {
        document.getElementById('marks-slider').value = val;
        document.getElementById('marks-input').value = val;
        document.getElementById('marks-val-display').innerText = val;
    } else if (type === 'hours') {
        document.getElementById('hours-slider').value = val;
        document.getElementById('hours-input').value = val;
        document.getElementById('hours-val-display').innerText = `${val} hrs`;
    } else if (type === 'attendance') {
        document.getElementById('attendance-slider').value = val;
        document.getElementById('attendance-input').value = val;
        document.getElementById('attendance-val-display').innerText = `${val}%`;
    }

    clearTimeout(predictDebounceTimer);
    predictDebounceTimer = setTimeout(runPrediction, 150);
}

function applyPreset(marks, hours, attendance) {
    syncInputs('marks', marks);
    syncInputs('hours', hours);
    syncInputs('attendance', attendance);
}

async function runPrediction() {
    const marks = document.getElementById('marks-input').value;
    const hours = document.getElementById('hours-input').value;
    const attendance = document.getElementById('attendance-input').value;
    const k = document.getElementById('knn-k-select').value;
    const kernel = document.getElementById('svm-kernel-select').value;

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                marks,
                hours,
                attendance,
                strategy: currentStrategy,
                k,
                kernel
            })
        });
        const data = await res.json();
        if (!data.success) return;

        // Render KNN Prediction
        const knnOutcomeEl = document.getElementById('knn-outcome-badge');
        if (data.knn.prediction.toLowerCase() === 'pass') {
            knnOutcomeEl.className = 'pred-outcome-badge outcome-pass';
            knnOutcomeEl.innerHTML = `<i class="fa-solid fa-circle-check"></i> PASS`;
        } else {
            knnOutcomeEl.className = 'pred-outcome-badge outcome-fail';
            knnOutcomeEl.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> FAIL`;
        }
        document.getElementById('knn-pass-prob').innerText = `${data.knn.pass_probability}%`;
        document.getElementById('knn-fail-prob').innerText = `${data.knn.fail_probability}%`;
        document.getElementById('knn-pass-bar').style.width = `${data.knn.pass_probability}%`;
        document.getElementById('knn-fail-bar').style.width = `${data.knn.fail_probability}%`;

        // Render SVM Prediction
        const svmOutcomeEl = document.getElementById('svm-outcome-badge');
        if (data.svm.prediction.toLowerCase() === 'pass') {
            svmOutcomeEl.className = 'pred-outcome-badge outcome-pass';
            svmOutcomeEl.innerHTML = `<i class="fa-solid fa-circle-check"></i> PASS`;
        } else {
            svmOutcomeEl.className = 'pred-outcome-badge outcome-fail';
            svmOutcomeEl.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> FAIL`;
        }
        document.getElementById('svm-pass-prob').innerText = `${data.svm.pass_probability}%`;
        document.getElementById('svm-fail-prob').innerText = `${data.svm.fail_probability}%`;
        document.getElementById('svm-pass-bar').style.width = `${data.svm.pass_probability}%`;
        document.getElementById('svm-fail-bar').style.width = `${data.svm.fail_probability}%`;

        // Consensus Banner
        const consensusEl = document.getElementById('consensus-text');
        if (data.agreement) {
            consensusEl.innerHTML = `<strong>Full Agreement:</strong> Both KNN and SVM predict the student will <strong>${data.knn.prediction.toUpperCase()}</strong>.`;
        } else {
            consensusEl.innerHTML = `<strong>Divergent Decision:</strong> KNN predicts <strong>${data.knn.prediction.toUpperCase()}</strong> (${data.knn.pass_probability}% pass prob) while SVM predicts <strong>${data.svm.prediction.toUpperCase()}</strong> (${data.svm.pass_probability}% pass prob).`;
        }

    } catch (err) {
        console.error('Error during prediction:', err);
    }
}

/* ==========================================================================
   4. Imputation Strategies Comparison Table
   ========================================================================== */
async function fetchImputationComparison() {
    try {
        const res = await fetch('/api/compare-imputations');
        const data = await res.json();

        const tbody = document.getElementById('imputation-comp-tbody');
        tbody.innerHTML = '';

        data.imputation_comparison.forEach(item => {
            const tr = document.createElement('tr');
            const isOptimal = item.strategy.toLowerCase() === 'mean';
            const recBadge = isOptimal
                ? `<span class="badge-pass"><i class="fa-solid fa-thumbs-up"></i> Recommended</span>`
                : (item.strategy.toLowerCase() === 'median' ? `<span class="badge-status">Good Alternative</span>` : `<span class="badge-fail">Least Recommended</span>`);

            tr.innerHTML = `
                <td><strong>${item.strategy}</strong></td>
                <td><code>Marks: ${item.imputed_marks} | Hours: ${item.imputed_hours} | Att: ${item.imputed_attendance}%</code></td>
                <td><strong>${item.knn_test_acc}%</strong></td>
                <td>${item.knn_cv_acc}%</td>
                <td><strong>${item.svm_test_acc}%</strong></td>
                <td>${item.svm_cv_acc}%</td>
                <td>${recBadge}</td>
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.error('Error fetching imputation comparison:', err);
    }
}

/* ==========================================================================
   5. Chart.js Visualizations
   ========================================================================== */
function initCharts() {
    const accCtx = document.getElementById('accuracyChart').getContext('2d');
    accuracyChart = new Chart(accCtx, {
        type: 'bar',
        data: {
            labels: ['Train Acc', 'Test Acc', '5-Fold CV'],
            datasets: [
                {
                    label: 'KNN',
                    data: [0, 0, 0],
                    backgroundColor: '#38bdf8',
                    borderRadius: 6,
                },
                {
                    label: 'SVM',
                    data: [0, 0, 0],
                    backgroundColor: '#f87171',
                    borderRadius: 6,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 110,
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#94a3b8', callback: val => `${val}%` }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f8fafc', font: { family: 'Plus Jakarta Sans', weight: '600' } } }
            }
        }
    });

    const metCtx = document.getElementById('metricsChart').getContext('2d');
    metricsChart = new Chart(metCtx, {
        type: 'bar',
        data: {
            labels: ['Precision', 'Recall', 'F1-Score'],
            datasets: [
                {
                    label: 'KNN',
                    data: [0, 0, 0],
                    backgroundColor: '#34d399',
                    borderRadius: 6,
                },
                {
                    label: 'SVM',
                    data: [0, 0, 0],
                    backgroundColor: '#c084fc',
                    borderRadius: 6,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 110,
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#94a3b8', callback: val => `${val}%` }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f8fafc', font: { family: 'Plus Jakarta Sans', weight: '600' } } }
            }
        }
    });
}

function updateCharts(data) {
    if (accuracyChart) {
        accuracyChart.data.datasets[0].data = [
            data.knn.train_accuracy,
            data.knn.test_accuracy,
            data.knn.cv_accuracy_mean
        ];
        accuracyChart.data.datasets[1].data = [
            data.svm.train_accuracy,
            data.svm.test_accuracy,
            data.svm.cv_accuracy_mean
        ];
        accuracyChart.update();
    }

    if (metricsChart) {
        metricsChart.data.datasets[0].data = [
            data.knn.precision,
            data.knn.recall,
            data.knn.f1_score
        ];
        metricsChart.data.datasets[1].data = [
            data.svm.precision,
            data.svm.recall,
            data.svm.f1_score
        ];
        metricsChart.update();
    }
}
