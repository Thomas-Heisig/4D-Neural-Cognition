// Real-time Analytics Module
// Provides spike rate histograms, network statistics, learning curves, and performance metrics

class NetworkAnalytics {
    constructor() {
        this.charts = {};
        this.data = {
            spikeRates: [],
            networkStats: [],
            learningCurve: [],
            performanceMetrics: []
        };
        this.colors = {
            primary: '#4a90e2',
            secondary: '#7b68ee',
            success: '#5cb85c',
            danger: '#d9534f',
            warning: '#f0ad4e'
        };
    }
    
    // Initialize spike rate histogram
    initSpikeRateHistogram(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        
        this.charts.spikeRate = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Spike Rate (spikes/step)',
                    data: [],
                    backgroundColor: this.colors.primary,
                    borderColor: this.colors.primary,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Spikes',
                            color: '#e0e0e0'
                        },
                        ticks: { color: '#a0a0a0' },
                        grid: { color: '#3a3a4e' }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time Step',
                            color: '#e0e0e0'
                        },
                        ticks: { color: '#a0a0a0' },
                        grid: { color: '#3a3a4e' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                }
            }
        });
    }
    
    // Initialize network statistics dashboard
    initNetworkStats(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        
        this.charts.networkStats = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Active Neurons',
                        data: [],
                        borderColor: this.colors.success,
                        backgroundColor: this.colors.success + '40',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Active Synapses',
                        data: [],
                        borderColor: this.colors.secondary,
                        backgroundColor: this.colors.secondary + '40',
                        fill: true,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count',
                            color: '#e0e0e0'
                        },
                        ticks: { color: '#a0a0a0' },
                        grid: { color: '#3a3a4e' }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time Step',
                            color: '#e0e0e0'
                        },
                        ticks: { color: '#a0a0a0' },
                        grid: { color: '#3a3a4e' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                }
            }
        });
    }
    
    // Initialize learning curves
    initLearningCurves(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        
        this.charts.learningCurve = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: this.colors.danger,
                        backgroundColor: this.colors.danger + '40',
                        yAxisID: 'y',
                        tension: 0.4
                    },
                    {
                        label: 'Accuracy',
                        data: [],
                        borderColor: this.colors.success,
                        backgroundColor: this.colors.success + '40',
                        yAxisID: 'y1',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Loss',
                            color: '#e0e0e0'
                        },
                        ticks: { color: '#a0a0a0' },
                        grid: { color: '#3a3a4e' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Accuracy',
                            color: '#e0e0e0'
                        },
                        ticks: { color: '#a0a0a0' },
                        grid: { drawOnChartArea: false }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch',
                            color: '#e0e0e0'
                        },
                        ticks: { color: '#a0a0a0' },
                        grid: { color: '#3a3a4e' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                }
            }
        });
    }
    
    // Initialize performance metrics dashboard
    initPerformanceMetrics(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        
        this.charts.performanceMetrics = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Stability'],
                datasets: [{
                    label: 'Current Model',
                    data: [0, 0, 0, 0, 0],
                    borderColor: this.colors.primary,
                    backgroundColor: this.colors.primary + '60',
                    pointBackgroundColor: this.colors.primary,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: this.colors.primary
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            stepSize: 0.2,
                            color: '#a0a0a0'
                        },
                        grid: { color: '#3a3a4e' },
                        pointLabels: { color: '#e0e0e0' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                }
            }
        });
    }
    
    // Update spike rate histogram with new data
    updateSpikeRates(step, spikeCount) {
        if (!this.charts.spikeRate) return;
        
        this.data.spikeRates.push({ step, count: spikeCount });
        
        // Keep only last 50 data points
        if (this.data.spikeRates.length > 50) {
            this.data.spikeRates.shift();
        }
        
        const labels = this.data.spikeRates.map(d => d.step);
        const data = this.data.spikeRates.map(d => d.count);
        
        this.charts.spikeRate.data.labels = labels;
        this.charts.spikeRate.data.datasets[0].data = data;
        this.charts.spikeRate.update('none'); // Update without animation for performance
    }
    
    // Update network statistics
    updateNetworkStats(step, neurons, synapses) {
        if (!this.charts.networkStats) return;
        
        this.data.networkStats.push({ step, neurons, synapses });
        
        // Keep only last 100 data points
        if (this.data.networkStats.length > 100) {
            this.data.networkStats.shift();
        }
        
        const labels = this.data.networkStats.map(d => d.step);
        const neuronData = this.data.networkStats.map(d => d.neurons);
        const synapseData = this.data.networkStats.map(d => d.synapses);
        
        this.charts.networkStats.data.labels = labels;
        this.charts.networkStats.data.datasets[0].data = neuronData;
        this.charts.networkStats.data.datasets[1].data = synapseData;
        this.charts.networkStats.update('none');
    }
    
    // Update learning curves
    updateLearningCurve(epoch, loss, accuracy) {
        if (!this.charts.learningCurve) return;
        
        this.data.learningCurve.push({ epoch, loss, accuracy });
        
        const labels = this.data.learningCurve.map(d => d.epoch);
        const lossData = this.data.learningCurve.map(d => d.loss);
        const accuracyData = this.data.learningCurve.map(d => d.accuracy);
        
        this.charts.learningCurve.data.labels = labels;
        this.charts.learningCurve.data.datasets[0].data = lossData;
        this.charts.learningCurve.data.datasets[1].data = accuracyData;
        this.charts.learningCurve.update('none');
    }
    
    // Update performance metrics
    updatePerformanceMetrics(metrics) {
        if (!this.charts.performanceMetrics) return;
        
        const data = [
            metrics.accuracy || 0,
            metrics.precision || 0,
            metrics.recall || 0,
            metrics.f1Score || 0,
            metrics.stability || 0
        ];
        
        this.charts.performanceMetrics.data.datasets[0].data = data;
        this.charts.performanceMetrics.update();
    }
    
    // Clear all data
    clearAllData() {
        this.data = {
            spikeRates: [],
            networkStats: [],
            learningCurve: [],
            performanceMetrics: []
        };
        
        // Clear charts
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.data) {
                chart.data.labels = [];
                chart.data.datasets.forEach(dataset => {
                    dataset.data = [];
                });
                chart.update();
            }
        });
    }
    
    // Export data as JSON
    exportData() {
        return JSON.stringify(this.data, null, 2);
    }
    
    // Dispose all charts
    dispose() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
        this.charts = {};
    }
}

// Export for use in main app
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NetworkAnalytics;
}
