// Application state and data
const appData = {
    projectInfo: {
        title: "Image Classification using Convolutional Neural Networks (CNNs)",
        authors: ["Akshat Jain (220905090)", "Amartya Singh (220905250)", "Mrityunjaya Sharma (220905444)"],
        institution: "Department of Computer Science and Engineering, Manipal Institute of Technology, Manipal"
    },
    
    architecture: {
        name: "VGG-inspired CNN",
        inputSize: [32, 32, 3],
        blocks: [
            { name: "Block 1", filters: 64, layers: 2, outputSize: [16, 16, 64] },
            { name: "Block 2", filters: 128, layers: 2, outputSize: [8, 8, 128] },
            { name: "Block 3", filters: 256, layers: 3, outputSize: [4, 4, 256] },
            { name: "Block 4", filters: 512, layers: 3, outputSize: [2, 2, 512] },
            { name: "Block 5", filters: 512, layers: 3, outputSize: [1, 1, 512] }
        ],
        classifier: [
            { type: "Linear", units: 4096 },
            { type: "Linear", units: 4096 },
            { type: "Linear", units: 10 }
        ]
    },
    
    dataset: {
        name: "CIFAR-10",
        classes: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        trainSamples: 45000,
        valSamples: 5000,
        testSamples: 10000,
        imageSize: [32, 32, 3],
        augmentation: ["Random Horizontal Flip", "Random Rotation (±10°)", "Random Crop with Padding", "Color Jitter"]
    },
    
    trainingParams: {
        optimizer: "Adam",
        learningRate: 0.001,
        batchSize: 128,
        epochs: 50,
        weightDecay: 1e-4,
        scheduler: "ReduceLROnPlateau",
        earlyStoppingPatience: 10
    },
    
    simulatedResults: {
        accuracy: 0.8745,
        precision: 0.8723,
        recall: 0.8745,
        f1Score: 0.8734,
        trainingEpochs: 35,
        bestValAcc: 0.8892
    },
    
    trainingHistory: {
        epochs: [1, 5, 10, 15, 20, 25, 30, 35],
        trainLoss: [2.1, 1.4, 0.9, 0.6, 0.4, 0.3, 0.25, 0.2],
        valLoss: [1.8, 1.2, 0.8, 0.65, 0.5, 0.45, 0.4, 0.38],
        trainAcc: [20, 45, 65, 75, 82, 86, 88, 90],
        valAcc: [25, 50, 68, 76, 83, 85, 87, 89]
    },
    
    confusionMatrix: {
        classes: ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        values: [
            [850, 12, 25, 8, 5, 2, 8, 15, 65, 10],
            [18, 920, 5, 15, 3, 8, 2, 5, 15, 9],
            [35, 8, 800, 45, 60, 25, 15, 10, 1, 1],
            [12, 20, 55, 780, 25, 85, 15, 6, 1, 1],
            [8, 5, 45, 35, 845, 15, 35, 12, 0, 0],
            [5, 15, 20, 90, 18, 825, 8, 19, 0, 0],
            [12, 8, 25, 28, 45, 15, 860, 7, 0, 0],
            [18, 12, 15, 15, 25, 35, 8, 870, 1, 1],
            [45, 25, 8, 5, 2, 1, 2, 2, 895, 15],
            [25, 35, 2, 8, 1, 2, 1, 5, 25, 896]
        ]
    },
    
    perClassMetrics: [
        { class: "airplane", precision: 0.85, recall: 0.85, f1: 0.85, support: 1000 },
        { class: "automobile", precision: 0.92, recall: 0.92, f1: 0.92, support: 1000 },
        { class: "bird", precision: 0.80, recall: 0.80, f1: 0.80, support: 1000 },
        { class: "cat", precision: 0.78, recall: 0.78, f1: 0.78, support: 1000 },
        { class: "deer", precision: 0.845, recall: 0.845, f1: 0.845, support: 1000 },
        { class: "dog", precision: 0.825, recall: 0.825, f1: 0.825, support: 1000 },
        { class: "frog", precision: 0.86, recall: 0.86, f1: 0.86, support: 1000 },
        { class: "horse", precision: 0.87, recall: 0.87, f1: 0.87, support: 1000 },
        { class: "ship", precision: 0.895, recall: 0.895, f1: 0.895, support: 1000 },
        { class: "truck", precision: 0.896, recall: 0.896, f1: 0.896, support: 1000 }
    ]
};

// Chart instances
let lossChart, accuracyChart, confusionChart, perClassChart;

// DOM elements
const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanels = document.querySelectorAll('.tab-panel');
const tooltip = document.getElementById('tooltip');

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeTabs();
    initializeTooltips();
    initializeCharts();
    populateMetricsTable();
    initializeInteractiveElements();
});

// Tab functionality
function initializeTabs() {
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Remove active class from all tabs and panels
    tabButtons.forEach(btn => btn.classList.remove('active'));
    tabPanels.forEach(panel => panel.classList.remove('active'));
    
    // Add active class to current tab and panel
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(tabName).classList.add('active');
    
    // Initialize charts when switching to training or evaluation tabs
    if (tabName === 'training') {
        setTimeout(() => {
            initializeTrainingCharts();
        }, 100);
    } else if (tabName === 'evaluation') {
        setTimeout(() => {
            initializeEvaluationCharts();
        }, 100);
    }
}

// Tooltip functionality
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
        element.addEventListener('mousemove', moveTooltip);
    });
}

function showTooltip(event) {
    const text = event.target.dataset.tooltip;
    tooltip.textContent = text;
    tooltip.classList.add('show');
    moveTooltip(event);
}

function hideTooltip() {
    tooltip.classList.remove('show');
}

function moveTooltip(event) {
    const rect = tooltip.getBoundingClientRect();
    let x = event.clientX + 10;
    let y = event.clientY + 10;
    
    // Prevent tooltip from going off-screen
    if (x + rect.width > window.innerWidth) {
        x = event.clientX - rect.width - 10;
    }
    if (y + rect.height > window.innerHeight) {
        y = event.clientY - rect.height - 10;
    }
    
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
}

// Chart initialization
function initializeCharts() {
    // Charts will be initialized when tabs are switched
}

function initializeTrainingCharts() {
    if (lossChart) lossChart.destroy();
    if (accuracyChart) accuracyChart.destroy();
    
    // Loss Chart
    const lossCtx = document.getElementById('lossChart');
    if (lossCtx) {
        lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: appData.trainingHistory.epochs,
                datasets: [
                    {
                        label: 'Training Loss',
                        data: appData.trainingHistory.trainLoss,
                        borderColor: '#1FB8CD',
                        backgroundColor: 'rgba(31, 184, 205, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Validation Loss',
                        data: appData.trainingHistory.valLoss,
                        borderColor: '#FFC185',
                        backgroundColor: 'rgba(255, 193, 133, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Loss'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }
    
    // Accuracy Chart
    const accuracyCtx = document.getElementById('accuracyChart');
    if (accuracyCtx) {
        accuracyChart = new Chart(accuracyCtx, {
            type: 'line',
            data: {
                labels: appData.trainingHistory.epochs,
                datasets: [
                    {
                        label: 'Training Accuracy',
                        data: appData.trainingHistory.trainAcc,
                        borderColor: '#B4413C',
                        backgroundColor: 'rgba(180, 65, 60, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Validation Accuracy',
                        data: appData.trainingHistory.valAcc,
                        borderColor: '#5D878F',
                        backgroundColor: 'rgba(93, 135, 143, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }
}

function initializeEvaluationCharts() {
    if (confusionChart) confusionChart.destroy();
    if (perClassChart) perClassChart.destroy();
    
    // Confusion Matrix (simplified as heatmap using bar chart)
    const confusionCtx = document.getElementById('confusionMatrix');
    if (confusionCtx) {
        // Calculate accuracy for each class (diagonal values)
        const classAccuracies = appData.confusionMatrix.values.map((row, i) => {
            const total = row.reduce((sum, val) => sum + val, 0);
            return (row[i] / total * 100).toFixed(1);
        });
        
        confusionChart = new Chart(confusionCtx, {
            type: 'bar',
            data: {
                labels: appData.confusionMatrix.classes,
                datasets: [{
                    label: 'Class Accuracy (%)',
                    data: classAccuracies,
                    backgroundColor: [
                        '#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F',
                        '#DB4545', '#D2BA4C', '#964325', '#944454', '#13343B'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    // Per-class metrics chart
    const perClassCtx = document.getElementById('perClassChart');
    if (perClassCtx) {
        perClassChart = new Chart(perClassCtx, {
            type: 'radar',
            data: {
                labels: appData.perClassMetrics.map(metric => metric.class),
                datasets: [
                    {
                        label: 'Precision',
                        data: appData.perClassMetrics.map(metric => metric.precision * 100),
                        borderColor: '#1FB8CD',
                        backgroundColor: 'rgba(31, 184, 205, 0.2)',
                        borderWidth: 2
                    },
                    {
                        label: 'Recall',
                        data: appData.perClassMetrics.map(metric => metric.recall * 100),
                        borderColor: '#FFC185',
                        backgroundColor: 'rgba(255, 193, 133, 0.2)',
                        borderWidth: 2
                    },
                    {
                        label: 'F1-Score',
                        data: appData.perClassMetrics.map(metric => metric.f1 * 100),
                        borderColor: '#B4413C',
                        backgroundColor: 'rgba(180, 65, 60, 0.2)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: {
                            display: true
                        },
                        suggestedMin: 70,
                        suggestedMax: 100
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }
}

// Populate metrics table
function populateMetricsTable() {
    const tableBody = document.getElementById('metricsTableBody');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    appData.perClassMetrics.forEach(metric => {
        const row = document.createElement('div');
        row.className = 'table-row';
        row.innerHTML = `
            <div>${metric.class}</div>
            <div>${(metric.precision * 100).toFixed(1)}%</div>
            <div>${(metric.recall * 100).toFixed(1)}%</div>
            <div>${(metric.f1 * 100).toFixed(1)}%</div>
            <div>${metric.support}</div>
        `;
        tableBody.appendChild(row);
    });
}

// Interactive elements
function initializeInteractiveElements() {
    // Add click handlers for class items
    const classItems = document.querySelectorAll('.class-item');
    classItems.forEach(item => {
        item.addEventListener('click', () => {
            const className = item.dataset.class;
            highlightClassMetrics(className);
        });
    });
    
    // Add click handlers for architecture components
    const architectureComponents = document.querySelectorAll('.layer, .conv-block, .classifier');
    architectureComponents.forEach(component => {
        component.addEventListener('click', () => {
            component.style.transform = 'translateY(-8px)';
            setTimeout(() => {
                component.style.transform = 'translateY(-4px)';
            }, 200);
        });
    });
}

function highlightClassMetrics(className) {
    const metric = appData.perClassMetrics.find(m => m.class === className);
    if (metric) {
        // Show a temporary highlight with the class metrics
        const highlight = document.createElement('div');
        highlight.className = 'class-highlight';
        highlight.innerHTML = `
            <strong>${metric.class.toUpperCase()}</strong><br>
            Precision: ${(metric.precision * 100).toFixed(1)}%<br>
            Recall: ${(metric.recall * 100).toFixed(1)}%<br>
            F1-Score: ${(metric.f1 * 100).toFixed(1)}%
        `;
        highlight.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--color-primary);
            color: var(--color-btn-primary-text);
            padding: 20px;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-lg);
            z-index: 1000;
            text-align: center;
            font-size: var(--font-size-base);
        `;
        
        document.body.appendChild(highlight);
        
        setTimeout(() => {
            document.body.removeChild(highlight);
        }, 2000);
    }
}

// Utility functions for responsive behavior
function handleResize() {
    if (lossChart) lossChart.resize();
    if (accuracyChart) accuracyChart.resize();
    if (confusionChart) confusionChart.resize();
    if (perClassChart) perClassChart.resize();
}

window.addEventListener('resize', handleResize);

// Educational features - Add keyboard navigation
document.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        const activeTab = document.querySelector('.tab-btn.active');
        const tabs = Array.from(tabButtons);
        const currentIndex = tabs.indexOf(activeTab);
        let newIndex;
        
        if (event.key === 'ArrowRight') {
            newIndex = (currentIndex + 1) % tabs.length;
        } else {
            newIndex = (currentIndex - 1 + tabs.length) % tabs.length;
        }
        
        const newTab = tabs[newIndex];
        const tabName = newTab.dataset.tab;
        switchTab(tabName);
    }
});

// Add focus indicators for keyboard navigation
tabButtons.forEach(button => {
    button.setAttribute('tabindex', '0');
    button.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            button.click();
        }
    });
});

// Performance monitoring
const performanceObserver = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
        if (entry.entryType === 'navigation') {
            console.log(`Page load time: ${entry.loadEventEnd - entry.loadEventStart}ms`);
        }
    }
});

if (typeof PerformanceObserver !== 'undefined') {
    performanceObserver.observe({ entryTypes: ['navigation'] });
}

// Export for debugging (optional)
if (typeof window !== 'undefined') {
    window.appData = appData;
    window.switchTab = switchTab;
}